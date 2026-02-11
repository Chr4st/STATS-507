"""Arbitrage logic: converts nowcasts into ranked trade ideas."""

from __future__ import annotations

import uuid
from datetime import datetime

from geoag.common.config import ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.common.schemas import (
    InstrumentLeg,
    RegionNowcast,
    TradeIdea,
    TradeType,
)
from geoag.common.timeutils import now_utc
from geoag.signals.risk import RiskEngine
from geoag.signals.sessions import SessionChecker

logger = get_logger("signals.arb_logic")


class ArbEngine:
    """Generate ranked trade ideas from regional nowcasts."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()
        self.risk_engine = RiskEngine(self.config)
        self.session_checker = SessionChecker(self.config)

    def generate_ideas(
        self,
        nowcasts: list[RegionNowcast],
        spot_prices: dict[str, float] | None = None,
    ) -> list[TradeIdea]:
        """Generate and rank trade ideas from nowcasts."""
        timestamp = now_utc()
        spot_prices = spot_prices or self._get_latest_prices()

        ideas: list[TradeIdea] = []

        # 1. Cross-region spreads
        ideas.extend(self._cross_region_spreads(nowcasts, spot_prices, timestamp))

        # 2. Commodity substitution trades
        ideas.extend(self._substitution_trades(nowcasts, spot_prices, timestamp))

        # 3. Volatility catalyst trades
        ideas.extend(self._volatility_trades(nowcasts, spot_prices, timestamp))

        # 4. Directional trades for extreme signals
        ideas.extend(self._directional_trades(nowcasts, spot_prices, timestamp))

        # Rank by expected edge
        ideas.sort(key=lambda x: x.expected_edge, reverse=True)

        # Top K
        top_k = self.config.settings.signal.top_k_ideas
        return ideas[:top_k]

    def _cross_region_spreads(
        self,
        nowcasts: list[RegionNowcast],
        spot_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeIdea]:
        """Cross-region spread: long beneficiary, short impacted."""
        ideas: list[TradeIdea] = []
        {nc.region_id: nc for nc in nowcasts}

        # Group regions by crop
        crop_regions: dict[str, list[RegionNowcast]] = {}
        for nc in nowcasts:
            region_info = self.config.region_map.get(nc.region_id)
            if region_info:
                crop = region_info.crop
                crop_regions.setdefault(crop, []).append(nc)

        for crop, regions in crop_regions.items():
            if len(regions) < 2:
                continue

            # Sort by standardized shock (worst to best)
            sorted_regions = sorted(
                regions,
                key=lambda r: r.yield_shock_mean / max(r.yield_shock_sigma, 0.1),
            )

            worst = sorted_regions[0]
            best = sorted_regions[-1]

            shock_diff = (
                best.yield_shock_mean / max(best.yield_shock_sigma, 0.1)
                - worst.yield_shock_mean / max(worst.yield_shock_sigma, 0.1)
            )

            if abs(shock_diff) < 0.3:
                continue

            # Find instruments
            worst_syms = self.config.instrument_for_region(worst.region_id)
            best_syms = self.config.instrument_for_region(best.region_id)

            if not worst_syms or not best_syms:
                continue

            short_sym = worst_syms[0]
            long_sym = best_syms[0]

            if short_sym == long_sym:
                continue

            all_syms = [long_sym, short_sym]
            risk_result = self.risk_engine.evaluate(all_syms, spot_prices)

            edge = self._compute_edge(
                abs(shock_diff), 0.0, 0.8 if not risk_result["risk_tags"] else 0.4,
                risk_result["pin_risk_score"],
            )

            worst_info = self.config.region_map.get(worst.region_id)
            best_info = self.config.region_map.get(best.region_id)
            worst_name = worst_info.name if worst_info else worst.region_id
            best_name = best_info.name if best_info else best.region_id

            ideas.append(
                TradeIdea(
                    id=str(uuid.uuid4())[:8],
                    timestamp=timestamp,
                    trade_type=TradeType.SPREAD,
                    instruments=[
                        InstrumentLeg(
                            symbol=long_sym,
                            direction="long",
                            weight=1.0,
                            instrument_name=self.config.instruments[long_sym].name,
                        ),
                        InstrumentLeg(
                            symbol=short_sym,
                            direction="short",
                            weight=1.0,
                            instrument_name=self.config.instruments[short_sym].name,
                        ),
                    ],
                    rationale=(
                        f"{crop.upper()} spread: {best_name} stable "
                        f"(shock {best.yield_shock_mean:+.2f}) vs "
                        f"{worst_name} stressed "
                        f"(shock {worst.yield_shock_mean:+.2f}). "
                        f"Drivers: {', '.join(worst.drivers[:3])}"
                    ),
                    expected_edge=round(edge, 4),
                    risk_tags=risk_result["risk_tags"],  # type: ignore[arg-type]
                    risk_notes=risk_result["risk_notes"],  # type: ignore[arg-type]
                    tradable_now=risk_result["tradable_now"],  # type: ignore[arg-type]
                    best_window=risk_result.get("best_window"),  # type: ignore[arg-type]
                    hedges=[
                        f"Hedge with {crop} ETF proxy",
                        f"Consider {crop} options spread for defined risk",
                    ],
                    pin_risk_score=risk_result["pin_risk_score"],  # type: ignore[arg-type]
                    confidence=min(best.confidence, worst.confidence),
                )
            )

        return ideas

    def _substitution_trades(
        self,
        nowcasts: list[RegionNowcast],
        spot_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeIdea]:
        """Commodity substitution trades: if wheat stressed, consider corn/soy."""
        ideas: list[TradeIdea] = []
        {nc.region_id: nc for nc in nowcasts}

        for commodity, subs in self.config.substitutions.items():
            # Find regions for this commodity with high stress
            stressed_regions = [
                nc for nc in nowcasts
                if self.config.region_map.get(nc.region_id)
                and self.config.region_map[nc.region_id].crop == commodity
                and abs(nc.stress_index) > 0.8
            ]

            if not stressed_regions:
                continue

            avg_stress = sum(abs(r.stress_index) for r in stressed_regions) / len(stressed_regions)

            for sub in subs:
                sub_commodity = sub.commodity
                # Find instruments for stressed and substitute
                stressed_syms = set()
                for r in stressed_regions:
                    stressed_syms.update(self.config.instrument_for_region(r.region_id))

                sub_syms = set()
                for nc in nowcasts:
                    ri = self.config.region_map.get(nc.region_id)
                    if ri and ri.crop == sub_commodity:
                        sub_syms.update(self.config.instrument_for_region(nc.region_id))

                if not stressed_syms or not sub_syms:
                    continue

                stressed_sym = sorted(stressed_syms)[0]
                sub_sym = sorted(sub_syms)[0]

                if stressed_sym == sub_sym:
                    continue

                all_syms = [stressed_sym, sub_sym]
                risk_result = self.risk_engine.evaluate(all_syms, spot_prices)

                edge = self._compute_edge(
                    avg_stress * sub.correlation * 0.5,
                    0.3,
                    0.6,
                    risk_result["pin_risk_score"],
                )

                ideas.append(
                    TradeIdea(
                        id=str(uuid.uuid4())[:8],
                        timestamp=timestamp,
                        trade_type=TradeType.HEDGE,
                        instruments=[
                            InstrumentLeg(
                                symbol=sub_sym,
                                direction="long",
                                weight=sub.correlation,
                                instrument_name=self.config.instruments.get(sub_sym, type("", (), {"name": sub_sym})).name,
                            ),
                            InstrumentLeg(
                                symbol=stressed_sym,
                                direction="short",
                                weight=1.0,
                                instrument_name=self.config.instruments.get(stressed_sym, type("", (), {"name": stressed_sym})).name,
                            ),
                        ],
                        rationale=(
                            f"Substitution: {commodity} stress ({avg_stress:.2f}) → "
                            f"rotate to {sub_commodity} "
                            f"(corr={sub.correlation:.2f}). {sub.note}"
                        ),
                        expected_edge=round(edge, 4),
                        risk_tags=risk_result["risk_tags"],  # type: ignore[arg-type]
                        risk_notes=risk_result["risk_notes"],  # type: ignore[arg-type]
                        tradable_now=risk_result["tradable_now"],  # type: ignore[arg-type]
                        best_window=risk_result.get("best_window"),  # type: ignore[arg-type]
                        hedges=[f"Use {sub_commodity} options for defined risk"],
                        pin_risk_score=risk_result["pin_risk_score"],  # type: ignore[arg-type]
                        confidence=0.5,
                    )
                )

        return ideas

    def _volatility_trades(
        self,
        nowcasts: list[RegionNowcast],
        spot_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeIdea]:
        """Volatility catalyst trades: high sigma + upcoming catalysts."""
        ideas: list[TradeIdea] = []

        # Check for upcoming catalysts
        from geoag.signals.macro import _build_catalyst_calendar

        catalysts = _build_catalyst_calendar(self.config, timestamp)
        if not catalysts:
            return ideas

        # Find regions with high yield_shock_sigma
        high_vol_regions = [
            nc for nc in nowcasts if nc.yield_shock_sigma > 0.5
        ]

        for nc in high_vol_regions:
            region_info = self.config.region_map.get(nc.region_id)
            if not region_info:
                continue

            syms = self.config.instrument_for_region(nc.region_id)
            if not syms:
                continue

            sym = syms[0]
            risk_result = self.risk_engine.evaluate([sym], spot_prices)

            nearest_catalyst = catalysts[0]
            catalyst_score = 0.5 if nearest_catalyst.days_until <= 7 else 0.2

            edge = self._compute_edge(
                nc.yield_shock_sigma * 0.5,
                catalyst_score,
                0.7,
                risk_result["pin_risk_score"],
            )

            ideas.append(
                TradeIdea(
                    id=str(uuid.uuid4())[:8],
                    timestamp=timestamp,
                    trade_type=TradeType.VOLATILITY,
                    instruments=[
                        InstrumentLeg(
                            symbol=sym,
                            direction="long",
                            weight=1.0,
                            instrument_name=self.config.instruments[sym].name,
                        ),
                    ],
                    rationale=(
                        f"Vol catalyst: {region_info.name} sigma={nc.yield_shock_sigma:.2f}, "
                        f"upcoming {nearest_catalyst.name} in {nearest_catalyst.days_until}d. "
                        f"Consider straddle/strangle on {sym}."
                    ),
                    expected_edge=round(edge, 4),
                    risk_tags=risk_result["risk_tags"],  # type: ignore[arg-type]
                    risk_notes=risk_result["risk_notes"],  # type: ignore[arg-type]
                    tradable_now=risk_result["tradable_now"],  # type: ignore[arg-type]
                    best_window=risk_result.get("best_window"),  # type: ignore[arg-type]
                    hedges=["Define risk with put/call spread instead of naked straddle"],
                    pin_risk_score=risk_result["pin_risk_score"],  # type: ignore[arg-type]
                    confidence=nc.confidence * 0.8,
                )
            )

        return ideas

    def _directional_trades(
        self,
        nowcasts: list[RegionNowcast],
        spot_prices: dict[str, float],
        timestamp: datetime,
    ) -> list[TradeIdea]:
        """Directional trades for extreme signals."""
        ideas: list[TradeIdea] = []

        for nc in nowcasts:
            if abs(nc.stress_index) < 1.2:
                continue

            region_info = self.config.region_map.get(nc.region_id)
            if not region_info:
                continue

            syms = self.config.instrument_for_region(nc.region_id)
            if not syms:
                continue

            sym = syms[0]
            direction = "long" if nc.stress_index > 0 else "short"
            risk_result = self.risk_engine.evaluate([sym], spot_prices)

            edge = self._compute_edge(
                abs(nc.stress_index) * 0.4,
                0.1,
                0.7,
                risk_result["pin_risk_score"],
            )

            ideas.append(
                TradeIdea(
                    id=str(uuid.uuid4())[:8],
                    timestamp=timestamp,
                    trade_type=TradeType.DIRECTIONAL,
                    instruments=[
                        InstrumentLeg(
                            symbol=sym,
                            direction=direction,
                            weight=1.0,
                            instrument_name=self.config.instruments[sym].name,
                        ),
                    ],
                    rationale=(
                        f"Extreme stress in {region_info.name} "
                        f"(idx={nc.stress_index:+.2f}, conf={nc.confidence:.2f}). "
                        f"Drivers: {', '.join(nc.drivers[:3])}. "
                        f"{'Supply risk → bullish' if nc.stress_index > 0 else 'Oversupply signal → bearish'} "
                        f"{region_info.crop}."
                    ),
                    expected_edge=round(edge, 4),
                    risk_tags=risk_result["risk_tags"],  # type: ignore[arg-type]
                    risk_notes=risk_result["risk_notes"],  # type: ignore[arg-type]
                    tradable_now=risk_result["tradable_now"],  # type: ignore[arg-type]
                    best_window=risk_result.get("best_window"),  # type: ignore[arg-type]
                    hedges=[
                        f"Hedge with {region_info.crop} options or correlated ETF",
                        f"Proxy ETF: {self.config.instruments[sym].proxy_etf}",
                    ],
                    pin_risk_score=risk_result["pin_risk_score"],  # type: ignore[arg-type]
                    confidence=nc.confidence,
                )
            )

        return ideas

    def _compute_edge(
        self,
        shock_diff: float,
        catalyst_score: float,
        liquidity_score: float,
        risk_penalty: float,
    ) -> float:
        """Compute expected edge score."""
        w = self.config.settings.signal.ranking_weights
        return (
            w.shock_diff * abs(shock_diff)
            + w.catalyst * catalyst_score
            + w.liquidity * liquidity_score
            - w.risk_penalty * risk_penalty
        )

    def _get_latest_prices(self) -> dict[str, float]:
        """Load latest close prices from data lake."""
        import pandas as pd

        from geoag.common.config import DATA_LAKE_DIR

        prices: dict[str, float] = {}
        for sym in self.config.instruments:
            path = DATA_LAKE_DIR / f"prices_{sym}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if not df.empty:
                    prices[sym] = float(df.iloc[-1]["close"])
        return prices
