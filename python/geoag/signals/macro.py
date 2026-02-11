"""Macro indicator computation from nowcasts."""

from __future__ import annotations

from datetime import datetime

from geoag.common.config import ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.common.schemas import CatalystEvent, ImpactLevel, MacroIndicators, RegionNowcast
from geoag.common.timeutils import now_utc

logger = get_logger("signals.macro")


def compute_macro(
    nowcasts: list[RegionNowcast],
    config: ConfigStore | None = None,
) -> MacroIndicators:
    """Compute global macro indicators from per-region nowcasts.

    Indicators:
    - global_crop_stress_nowcast: production-weighted average stress
    - export_risk_index: stress × export share × logistics placeholder
    - food_inflation_pressure_proxy: stress in staple basket
    - volatility_catalyst_calendar: upcoming events
    """
    cfg = config or get_config()
    timestamp = now_utc()

    # Build region→nowcast map
    nc_map = {nc.region_id: nc for nc in nowcasts}

    # Global crop stress (production-weighted)
    global_stress = _weighted_stress(nc_map, cfg.production_shares)

    # Export risk
    export_risk = _weighted_stress(nc_map, cfg.export_shares, scale_by_shock=True)

    # Food inflation pressure (equal-weight staples)
    staple_commodities = ["corn", "wheat", "soy"]
    inflation_pressure = _staple_inflation_pressure(nc_map, cfg, staple_commodities)

    # Catalyst calendar
    catalysts = _build_catalyst_calendar(cfg, timestamp)

    return MacroIndicators(
        timestamp=timestamp,
        global_crop_stress_nowcast=round(global_stress, 4),
        export_risk_index=round(export_risk, 4),
        food_inflation_pressure_proxy=round(inflation_pressure, 4),
        volatility_catalyst_calendar=catalysts,
    )


def _weighted_stress(
    nc_map: dict[str, RegionNowcast],
    shares: dict[str, dict[str, float]],
    scale_by_shock: bool = False,
) -> float:
    """Compute weighted stress across all commodities and regions."""
    total = 0.0
    weight_sum = 0.0

    for _commodity, region_shares in shares.items():
        for region_id, share in region_shares.items():
            if region_id == "rest_of_world":
                continue
            nc = nc_map.get(region_id)
            if nc is None:
                continue
            stress = abs(nc.stress_index)
            if scale_by_shock:
                stress *= abs(nc.yield_shock_mean) / max(nc.yield_shock_sigma, 0.1)
            total += stress * share
            weight_sum += share

    return total / max(weight_sum, 1e-8)


def _staple_inflation_pressure(
    nc_map: dict[str, RegionNowcast],
    cfg: ConfigStore,
    staple_commodities: list[str],
) -> float:
    """Average stress across staple commodity regions."""
    stresses: list[float] = []

    for commodity in staple_commodities:
        prod_shares = cfg.production_shares.get(commodity, {})
        for region_id, share in prod_shares.items():
            if region_id == "rest_of_world":
                continue
            nc = nc_map.get(region_id)
            if nc is not None:
                stresses.append(abs(nc.stress_index) * share)

    return sum(stresses) / max(len(stresses), 1)


def _build_catalyst_calendar(
    cfg: ConfigStore,
    now: datetime,
) -> list[CatalystEvent]:
    """Build list of upcoming catalysts within 30 days."""
    events: list[CatalystEvent] = []
    now_date = now.date()

    for entry in cfg.report_calendar:
        try:
            event_date = datetime.strptime(entry.date, "%Y-%m-%d").date()
        except ValueError:
            continue

        days_until = (event_date - now_date).days
        if 0 <= days_until <= 30:
            events.append(
                CatalystEvent(
                    date=entry.date,
                    name=entry.name,
                    agency=entry.agency,
                    impact=ImpactLevel(entry.impact),
                    commodities=entry.commodities,
                    days_until=days_until,
                )
            )

    events.sort(key=lambda e: e.days_until)
    return events
