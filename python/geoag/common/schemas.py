"""Pydantic schemas for all data objects flowing through the system."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TradeType(StrEnum):
    SPREAD = "spread"
    HEDGE = "hedge"
    DIRECTIONAL = "directional"
    VOLATILITY = "volatility"


class RiskTag(StrEnum):
    PIN_RISK = "PIN_RISK"
    SESSION_CLOSED = "SESSION_CLOSED"
    LOW_LIQUIDITY = "LOW_LIQUIDITY"
    SPEC_MISMATCH = "SPEC_MISMATCH"


class ImpactLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ---------------------------------------------------------------------------
# Region nowcast
# ---------------------------------------------------------------------------
class RegionNowcast(BaseModel):
    """Per-region nowcast output at time t."""

    region_id: str
    region_name: str
    timestamp: datetime
    stress_index: float = Field(description="Z-scored anomaly vs seasonal baseline")
    growth_index: float = Field(description="Phenology proxy 0-1")
    yield_shock_mean: float = Field(description="Yield shock distribution mean")
    yield_shock_sigma: float = Field(ge=0.0, description="Yield shock distribution sigma")
    confidence: float = Field(ge=0.0, le=1.0)
    drivers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Macro indicators
# ---------------------------------------------------------------------------
class MacroIndicators(BaseModel):
    """Global macro indicators computed daily."""

    timestamp: datetime
    global_crop_stress_nowcast: float
    export_risk_index: float
    food_inflation_pressure_proxy: float
    volatility_catalyst_calendar: list[CatalystEvent] = Field(default_factory=list)


class CatalystEvent(BaseModel):
    """Upcoming catalyst event."""

    date: str
    name: str
    agency: str
    impact: ImpactLevel
    commodities: list[str]
    days_until: int


# Re-declare MacroIndicators so CatalystEvent is defined first
MacroIndicators.model_rebuild()


# ---------------------------------------------------------------------------
# Instrument leg
# ---------------------------------------------------------------------------
class InstrumentLeg(BaseModel):
    """Single leg of a trade idea."""

    symbol: str
    direction: str = Field(description="'long' or 'short'")
    weight: float = 1.0
    instrument_name: str = ""


# ---------------------------------------------------------------------------
# Trade idea
# ---------------------------------------------------------------------------
class TradeIdea(BaseModel):
    """Ranked trade idea produced by the signal engine."""

    id: str
    timestamp: datetime
    trade_type: TradeType
    instruments: list[InstrumentLeg]
    rationale: str
    expected_edge: float = Field(description="Numeric score (not dollars)")
    risk_notes: list[str] = Field(default_factory=list)
    risk_tags: list[RiskTag] = Field(default_factory=list)
    tradable_now: bool = False
    best_window: str | None = None
    hedges: list[str] = Field(default_factory=list)
    pin_risk_score: float = Field(ge=0.0, le=1.0, default=0.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


# ---------------------------------------------------------------------------
# WebSocket message envelope
# ---------------------------------------------------------------------------
class WSMessage(BaseModel):
    """Envelope for all WebSocket messages."""

    type: str  # "macro", "regions", "trade_ideas", "heartbeat"
    timestamp: datetime
    data: dict | list | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime
    demo_mode: bool = True
