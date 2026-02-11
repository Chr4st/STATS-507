"""Inference service: loads trained model and produces nowcasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from geoag.common.config import DATA_LAKE_DIR, ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.common.schemas import RegionNowcast
from geoag.common.timeutils import now_utc
from geoag.model.model import build_model
from geoag.model.uncertainty import compute_confidence, mc_predict

logger = get_logger("model.infer")

MODELS_DIR = DATA_LAKE_DIR / "models"


class InferenceService:
    """Loads trained model and produces per-region nowcasts."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()
        mc = self.config.settings.model
        self.model = build_model(
            embedding_dim=self.config.settings.embedding.dim,
            weather_dim=8,
            hidden_dim=mc.hidden_dim,
            num_layers=mc.num_layers,
            dropout=mc.dropout,
        )
        self._load_weights()

    def _load_weights(self) -> None:
        """Load trained weights if available."""
        weights_path = MODELS_DIR / "nowcast_model.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded model weights from %s", weights_path)
        else:
            logger.warning("No trained weights found at %s — using random initialization", weights_path)

    def infer(self, features_df: pd.DataFrame) -> list[RegionNowcast]:
        """Run inference on latest features, producing per-region nowcasts."""
        config = self.config
        mc_cfg = config.settings.model
        dim = config.settings.embedding.dim
        seq_len = config.settings.feature.sequence_length
        timestamp = now_utc()

        nowcasts: list[RegionNowcast] = []

        for region_id, group in features_df.groupby("region_id"):
            region_info = config.region_map.get(str(region_id))
            if region_info is None:
                continue

            group = group.sort_values("date").tail(seq_len)
            n = len(group)

            # Extract tensors
            e_img = self._extract_embedding(group, "e_img", dim, n, seq_len)
            e_loc = self._extract_embedding(group, "e_loc", dim, n, seq_len)
            delta_e = self._extract_embedding(group, "delta_e", dim, n, seq_len)
            weather = self._extract_weather(group, n, seq_len)
            cloud_cover = self._extract_cloud(group, n, seq_len)

            # MC dropout inference
            mc_results = mc_predict(
                self.model, e_img, e_loc, delta_e, weather,
                n_samples=mc_cfg.mc_samples,
            )

            stress = float(mc_results["stress_index"]["mean"][0])
            growth = float(mc_results["growth_index"]["mean"][0])
            ym = float(mc_results["yield_shock_mean"]["mean"][0])
            yls = float(mc_results["yield_shock_log_sigma"]["mean"][0])
            sigma = float(torch.exp(torch.tensor(yls)).clamp(0.01, 10.0))

            # Uncertainty from MC std
            mc_std = float(mc_results["stress_index"]["std"][0])
            missingness = torch.tensor([max(0.0, 1.0 - n / seq_len)])
            cloud_mean = torch.tensor([float(cloud_cover.mean())])
            mc_std_t = torch.tensor([mc_std])

            cc = mc_cfg.confidence
            confidence = float(
                compute_confidence(mc_std_t, missingness, cloud_mean, cc.a, cc.b, cc.c, cc.d)[0]
            )

            # Determine drivers
            drivers = self._identify_drivers(group, stress, growth)

            nowcasts.append(
                RegionNowcast(
                    region_id=str(region_id),
                    region_name=region_info.name,
                    timestamp=timestamp,
                    stress_index=round(stress, 4),
                    growth_index=round(max(0, min(1, growth)), 4),
                    yield_shock_mean=round(ym, 4),
                    yield_shock_sigma=round(sigma, 4),
                    confidence=round(max(0, min(1, confidence)), 4),
                    drivers=drivers,
                )
            )

        logger.info("Produced %d nowcasts", len(nowcasts))
        return nowcasts

    def _extract_embedding(
        self, df: pd.DataFrame, prefix: str, dim: int, n: int, seq_len: int
    ) -> torch.Tensor:
        """Extract embedding columns and pad to seq_len."""
        cols = [f"{prefix}_{i}" for i in range(dim)]
        available = [c for c in cols if c in df.columns]
        if not available:
            arr = np.zeros((n, dim), dtype=np.float32)
        else:
            arr = df[available].values.astype(np.float32)
            if arr.shape[1] < dim:
                pad = np.zeros((n, dim - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])

        if n < seq_len:
            pad = np.zeros((seq_len - n, dim), dtype=np.float32)
            arr = np.vstack([pad, arr])

        return torch.from_numpy(arr).unsqueeze(0)  # (1, T, D)

    def _extract_weather(
        self, df: pd.DataFrame, n: int, seq_len: int
    ) -> torch.Tensor:
        """Extract weather features."""
        weather_cols = [
            "gdd", "rainfall_anomaly", "heat_stress_days", "drought_proxy",
            "temp_max", "temp_min", "precip_mm", "soil_moisture",
        ]
        available = [c for c in weather_cols if c in df.columns]
        w_dim = len(weather_cols)
        if not available:
            arr = np.zeros((n, w_dim), dtype=np.float32)
        else:
            arr = df[available].values.astype(np.float32)
            if arr.shape[1] < w_dim:
                pad = np.zeros((n, w_dim - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])

        if n < seq_len:
            pad = np.zeros((seq_len - n, w_dim), dtype=np.float32)
            arr = np.vstack([pad, arr])

        return torch.from_numpy(arr).unsqueeze(0)  # (1, T, 8)

    def _extract_cloud(
        self, df: pd.DataFrame, n: int, seq_len: int
    ) -> torch.Tensor:
        """Extract cloud cover."""
        if "cloud_cover" in df.columns:
            arr = df["cloud_cover"].values.astype(np.float32)
        else:
            arr = np.zeros(n, dtype=np.float32)

        if n < seq_len:
            arr = np.concatenate([np.zeros(seq_len - n, dtype=np.float32), arr])

        return torch.from_numpy(arr)

    def _identify_drivers(
        self, df: pd.DataFrame, stress: float, growth: float
    ) -> list[str]:
        """Identify top contributing factors for the nowcast."""
        drivers: list[str] = []
        if df.empty:
            return ["insufficient_data"]

        last = df.iloc[-1]

        # Check weather anomalies
        if "heat_stress_days" in last.index and float(last["heat_stress_days"]) > 0:
            drivers.append("heat_stress")
        if "drought_proxy" in last.index and float(last["drought_proxy"]) > 0.5:
            drivers.append("drought_risk")
        if "rainfall_anomaly" in last.index and float(last["rainfall_anomaly"]) < -3:
            drivers.append("low_rainfall")
        if "rainfall_anomaly" in last.index and float(last["rainfall_anomaly"]) > 10:
            drivers.append("excess_rainfall")
        if "gdd" in last.index and float(last["gdd"]) > 15:
            drivers.append("high_gdd")
        if abs(stress) > 1.0:
            drivers.append("embedding_anomaly")
        if growth < 0.3:
            drivers.append("low_greenness")

        if not drivers:
            drivers.append("normal_conditions")

        return drivers[:5]


def run_inference() -> list[RegionNowcast]:
    """Convenience entry point."""
    config = get_config()
    svc = InferenceService(config)
    features_path = DATA_LAKE_DIR / "features.parquet"
    if not features_path.exists():
        logger.error("No features.parquet found — run feature service first")
        return []
    df = pd.read_parquet(features_path)
    return svc.infer(df)


if __name__ == "__main__":
    from geoag.common.logging import setup_logging

    setup_logging("INFO")
    nowcasts = run_inference()
    for nc in nowcasts:
        print(nc.model_dump_json(indent=2))
