"""Risk engine: pin risk, spec mismatch, liquidity, session checks."""

from __future__ import annotations

from datetime import datetime

from geoag.common.config import ConfigStore, InstrumentSpec, get_config
from geoag.common.logging import get_logger
from geoag.common.schemas import RiskTag
from geoag.common.timeutils import now_utc
from geoag.signals.sessions import SessionChecker

logger = get_logger("signals.risk")


class RiskEngine:
    """Evaluates risk for trade ideas."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()
        self.session_checker = SessionChecker(self.config)

    def evaluate(
        self,
        symbols: list[str],
        spot_prices: dict[str, float] | None = None,
        at: datetime | None = None,
    ) -> dict[str, object]:
        """Evaluate risk for a set of instruments.

        Returns:
            {
                "risk_tags": [RiskTag, ...],
                "risk_notes": [str, ...],
                "tradable_now": bool,
                "best_window": str | None,
                "pin_risk_score": float,
            }
        """
        check_time = at or now_utc()
        spot_prices = spot_prices or {}

        risk_tags: list[RiskTag] = []
        risk_notes: list[str] = []
        all_tradable = True
        best_window: str | None = None
        max_pin_risk = 0.0

        # Per-instrument checks
        specs: list[InstrumentSpec] = []
        for sym in symbols:
            spec = self.config.instruments.get(sym)
            if spec is None:
                risk_notes.append(f"Unknown instrument: {sym}")
                continue
            specs.append(spec)

            # Session check
            sess_result = self.session_checker.check(sym, check_time)
            if not sess_result["tradable_now"]:
                all_tradable = False
                if RiskTag.SESSION_CLOSED not in risk_tags:
                    risk_tags.append(RiskTag.SESSION_CLOSED)
                risk_notes.append(f"{sym} session closed ({sess_result['session_type']})")
                if best_window is None:
                    best_window = sess_result.get("best_window")  # type: ignore[assignment]

            # Pin risk check
            pin = self._check_pin_risk(sym, spot_prices.get(sym))
            if pin > 0:
                max_pin_risk = max(max_pin_risk, pin)
                if pin > 0.5 and RiskTag.PIN_RISK not in risk_tags:
                    risk_tags.append(RiskTag.PIN_RISK)
                    risk_notes.append(
                        f"{sym} near OI peak (pin risk {pin:.2f})"
                    )

        # Spec mismatch check (cross-leg)
        if len(specs) > 1:
            mismatch_notes = self._check_spec_mismatch(specs, symbols)
            if mismatch_notes:
                risk_tags.append(RiskTag.SPEC_MISMATCH)
                risk_notes.extend(mismatch_notes)

        # Liquidity heuristic (demo: flag if volume is low)
        for sym in symbols:
            if self._is_low_liquidity(sym):
                if RiskTag.LOW_LIQUIDITY not in risk_tags:
                    risk_tags.append(RiskTag.LOW_LIQUIDITY)
                risk_notes.append(f"{sym} may have low liquidity")

        return {
            "risk_tags": risk_tags,
            "risk_notes": risk_notes,
            "tradable_now": all_tradable,
            "best_window": best_window,
            "pin_risk_score": round(max_pin_risk, 4),
        }

    def _check_pin_risk(self, symbol: str, spot: float | None) -> float:
        """Check pin risk based on OI peak config and spot price.

        Returns score 0–1.
        """
        peak_cfg = self.config.oi_peaks.get(symbol)
        if peak_cfg is None or spot is None:
            return 0.0

        threshold_pct = self.config.settings.signal.pin_risk_threshold_pct / 100.0

        # Check proximity to OI peaks
        max_score = 0.0
        for strike, weight in zip(peak_cfg.strikes, peak_cfg.oi_weights, strict=False):
            distance_pct = abs(spot - strike) / strike
            if distance_pct < threshold_pct:
                # Score based on distance (closer = higher) and OI weight
                proximity = 1.0 - (distance_pct / threshold_pct)
                score = proximity * weight * 3.0  # Scale up
                max_score = max(max_score, min(1.0, score))

        return max_score

    def _check_spec_mismatch(
        self, specs: list[InstrumentSpec], symbols: list[str]
    ) -> list[str]:
        """Check for material differences between legs."""
        notes: list[str] = []

        settlements = set(s.settlement for s in specs)
        if len(settlements) > 1:
            notes.append(
                f"Settlement mismatch: {', '.join(f'{sym}={s.settlement}' for sym, s in zip(symbols, specs, strict=False))}"
            )

        exercise_styles = set(s.exercise_style for s in specs)
        if len(exercise_styles) > 1:
            notes.append(
                f"Exercise style mismatch: {', '.join(f'{sym}={s.exercise_style}' for sym, s in zip(symbols, specs, strict=False))}"
            )

        currencies = set(s.currency for s in specs)
        if len(currencies) > 1:
            notes.append(
                f"Currency mismatch: {', '.join(f'{sym}={s.currency}' for sym, s in zip(symbols, specs, strict=False))}"
            )

        timezones = set(s.timezone for s in specs)
        if len(timezones) > 1:
            notes.append(
                f"Timezone mismatch: {', '.join(f'{sym}={s.timezone}' for sym, s in zip(symbols, specs, strict=False))}"
            )

        return notes

    def _is_low_liquidity(self, symbol: str) -> bool:
        """Heuristic: ETFs treated as less liquid for this domain."""
        spec = self.config.instruments.get(symbol)
        if spec is None:
            return True
        # In demo, flag non-futures as potentially lower liquidity for ag trades
        return spec.exchange == "NYSE" and spec.contract_size == 1
