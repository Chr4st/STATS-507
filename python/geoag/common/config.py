"""Configuration loader: reads YAML configs and provides typed access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]  # python/geoag/common -> repo root
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_SAMPLES_DIR = REPO_ROOT / "data_samples"
DATA_LAKE_DIR = REPO_ROOT / "data_lake"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_geojson(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Typed config models
# ---------------------------------------------------------------------------
class EmbeddingConfig(BaseModel):
    backend: str = "stub"
    dim: int = 512
    tiles_per_region: int = 1
    satclip_weights: str = "models/satclip_vit16.pth"


class FeatureConfig(BaseModel):
    sequence_length: int = 14
    seasonal_window: int = 30
    refresh_interval_s: int = 10


class ConfidenceCoeffs(BaseModel):
    a: float = 3.0
    b: float = 2.0
    c: float = 1.5
    d: float = 1.0


class ModelConfig(BaseModel):
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.15
    mc_samples: int = 10
    confidence: ConfidenceCoeffs = Field(default_factory=ConfidenceCoeffs)


class RankingWeights(BaseModel):
    shock_diff: float = 0.4
    catalyst: float = 0.2
    liquidity: float = 0.25
    risk_penalty: float = 0.15


class SignalConfig(BaseModel):
    ranking_weights: RankingWeights = Field(default_factory=RankingWeights)
    top_k_ideas: int = 10
    pin_risk_threshold_pct: float = 1.5
    pin_risk_dte_threshold: int = 5


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8777
    ws_path: str = "/ws"
    heartbeat_interval_s: int = 5


class PubSubConfig(BaseModel):
    backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"


class AppConfig(BaseModel):
    name: str = "GeoAg Arb Terminal"
    version: str = "0.1.0"
    timezone: str = "America/Detroit"
    seed: int = 42
    demo_mode: bool = True


class SessionSpec(BaseModel):
    electronic_open: str
    electronic_close: str
    pit_open: str
    pit_close: str
    break_start: str | None = None
    break_end: str | None = None


class InstrumentSpec(BaseModel):
    name: str
    exchange: str
    currency: str
    tick_size: float
    tick_value: float
    contract_size: int | float
    settlement: str
    exercise_style: str
    last_trade_day_offset: int
    front_months: list[str] = Field(default_factory=list)
    timezone: str
    session: SessionSpec
    regions: list[str] = Field(default_factory=list)
    proxy_etf: str = ""


class RegionInfo(BaseModel):
    id: str
    name: str
    crop: str
    country: str
    centroid_lat: float
    centroid_lon: float


class CatalystEntry(BaseModel):
    date: str
    name: str
    agency: str
    impact: str
    commodities: list[str]


class OIPeak(BaseModel):
    strikes: list[float]
    oi_weights: list[float]
    expiry: str


class SubstitutionEntry(BaseModel):
    commodity: str
    correlation: float
    note: str


# ---------------------------------------------------------------------------
# Aggregated settings
# ---------------------------------------------------------------------------
class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    pubsub: PubSubConfig = Field(default_factory=PubSubConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class ConfigStore:
    """Central config store — loads all configs once."""

    def __init__(self, configs_dir: Path | None = None) -> None:
        self._dir = configs_dir or CONFIGS_DIR
        self._raw_settings = _load_yaml(self._dir / "settings.yaml")
        self.settings = Settings(**self._raw_settings)

        # Instruments
        raw_instr = _load_yaml(self._dir / "instruments.yaml")
        self.instruments: dict[str, InstrumentSpec] = {
            k: InstrumentSpec(**v) for k, v in raw_instr.get("instruments", {}).items()
        }

        # Regions
        raw_geo = _load_geojson(self._dir / "regions.geojson")
        self.regions: list[RegionInfo] = [
            RegionInfo(**f["properties"]) for f in raw_geo.get("features", [])
        ]
        self.region_map: dict[str, RegionInfo] = {r.id: r for r in self.regions}

        # Calendars
        raw_cal = _load_yaml(self._dir / "calendars.yaml")
        self.holidays: dict[str, list[str]] = raw_cal.get("holidays", {})
        self.report_calendar: list[CatalystEntry] = [
            CatalystEntry(**e) for e in raw_cal.get("report_calendar", [])
        ]

        # Production & export shares
        self.production_shares: dict[str, dict[str, float]] = _load_yaml(
            self._dir / "production_shares.yaml"
        )
        self.export_shares: dict[str, dict[str, float]] = _load_yaml(
            self._dir / "export_shares.yaml"
        )

        # Substitution map
        raw_sub = _load_yaml(self._dir / "substitution_map.yaml")
        self.substitutions: dict[str, list[SubstitutionEntry]] = {}
        for commodity, entries in raw_sub.get("substitutions", {}).items():
            self.substitutions[commodity] = [SubstitutionEntry(**e) for e in entries]

        # Catalysts + OI peaks
        raw_cat = _load_yaml(self._dir / "catalysts.yaml")
        self.oi_peaks: dict[str, OIPeak] = {
            k: OIPeak(**v) for k, v in raw_cat.get("oi_peaks", {}).items()
        }
        self.weather_catalysts = raw_cat.get("weather_regime_catalysts", [])

    def get_holidays_for_exchange(self, exchange: str) -> list[str]:
        return self.holidays.get(exchange, [])

    def instrument_for_region(self, region_id: str) -> list[str]:
        """Return instrument symbols mapped to a region."""
        return [
            sym for sym, spec in self.instruments.items() if region_id in spec.regions
        ]


# Singleton-ish loader
_store: ConfigStore | None = None


def get_config(configs_dir: Path | None = None) -> ConfigStore:
    """Get or create the global ConfigStore."""
    global _store
    if _store is None:
        _store = ConfigStore(configs_dir)
    return _store


def reset_config() -> None:
    """Reset cached config (useful in tests)."""
    global _store
    _store = None
