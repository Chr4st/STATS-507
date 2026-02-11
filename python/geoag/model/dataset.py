"""Dataset construction for the nowcast time-series model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from geoag.common.config import DATA_LAKE_DIR, get_config
from geoag.common.logging import get_logger

logger = get_logger("model.dataset")


class NowcastDataset(Dataset):
    """Sliding-window dataset over region feature sequences.

    Each sample is a window of `seq_len` days of features for one region,
    with targets derived from the features themselves (self-supervised proxy
    for stress/growth/yield-shock).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        seq_len: int = 14,
        embedding_dim: int = 512,
    ) -> None:
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        # Weather feature columns
        self.weather_cols = [
            "gdd",
            "rainfall_anomaly",
            "heat_stress_days",
            "drought_proxy",
            "temp_max",
            "temp_min",
            "precip_mm",
            "soil_moisture",
        ]

        # Build sequences per region
        self.samples: list[dict[str, torch.Tensor]] = []
        self._build_samples(features_df)

    def _build_samples(self, df: pd.DataFrame) -> None:
        """Build sliding windows for each region."""
        for region_id, group in df.groupby("region_id"):
            group = group.sort_values("date").reset_index(drop=True)
            n = len(group)

            if n < self.seq_len:
                logger.debug(
                    "Region %s has %d rows < seq_len %d, using all",
                    region_id,
                    n,
                    self.seq_len,
                )
                # Pad with zeros
                pad_len = self.seq_len - n
                self._add_sample(group, pad_len, region_id)
            else:
                for start in range(n - self.seq_len + 1):
                    window = group.iloc[start : start + self.seq_len]
                    self._add_sample(window, 0, str(region_id))

    def _add_sample(
        self, window: pd.DataFrame, pad_len: int, region_id: str
    ) -> None:
        """Extract tensors from a window DataFrame."""
        dim = self.embedding_dim

        # Extract embedding columns
        e_img_cols = [f"e_img_{i}" for i in range(dim)]
        e_loc_cols = [f"e_loc_{i}" for i in range(dim)]
        delta_e_cols = [f"delta_e_{i}" for i in range(dim)]

        available_e_img = [c for c in e_img_cols if c in window.columns]
        available_e_loc = [c for c in e_loc_cols if c in window.columns]
        available_delta = [c for c in delta_e_cols if c in window.columns]
        available_weather = [c for c in self.weather_cols if c in window.columns]

        e_img = window[available_e_img].values.astype(np.float32) if available_e_img else np.zeros((len(window), dim), dtype=np.float32)
        e_loc = window[available_e_loc].values.astype(np.float32) if available_e_loc else np.zeros((len(window), dim), dtype=np.float32)
        delta_e = window[available_delta].values.astype(np.float32) if available_delta else np.zeros((len(window), dim), dtype=np.float32)
        weather = window[available_weather].values.astype(np.float32) if available_weather else np.zeros((len(window), len(self.weather_cols)), dtype=np.float32)

        cloud_cover = window["cloud_cover"].values.astype(np.float32) if "cloud_cover" in window.columns else np.zeros(len(window), dtype=np.float32)

        if pad_len > 0:
            e_img = np.vstack([np.zeros((pad_len, e_img.shape[1]), dtype=np.float32), e_img])
            e_loc = np.vstack([np.zeros((pad_len, e_loc.shape[1]), dtype=np.float32), e_loc])
            delta_e = np.vstack([np.zeros((pad_len, delta_e.shape[1]), dtype=np.float32), delta_e])
            weather = np.vstack([np.zeros((pad_len, weather.shape[1]), dtype=np.float32), weather])
            cloud_cover = np.concatenate([np.zeros(pad_len, dtype=np.float32), cloud_cover])

        # Targets: derived from last timestep features (self-supervised proxy)
        # stress ~ magnitude of delta_e (how much imagery changed)
        # growth ~ mean of green-related embedding dims
        # yield_shock ~ combination of weather stress indicators
        last_delta = delta_e[-1]
        last_weather = weather[-1]

        stress_target = float(np.linalg.norm(last_delta))
        growth_target = float(np.clip(np.mean(e_img[-1, :10]) + 0.5, 0, 1))
        yield_shock_mean_target = float(
            -last_weather[2] * 0.5 + last_weather[3] * 0.3 + stress_target * 0.2
        ) if last_weather.shape[0] > 3 else 0.0
        yield_shock_sigma_target = float(max(0.1, abs(stress_target) * 0.5))

        self.samples.append(
            {
                "e_img": torch.from_numpy(e_img),
                "e_loc": torch.from_numpy(e_loc),
                "delta_e": torch.from_numpy(delta_e),
                "weather": torch.from_numpy(weather),
                "cloud_cover": torch.from_numpy(cloud_cover),
                "region_id": region_id,
                "stress_target": torch.tensor(stress_target),
                "growth_target": torch.tensor(growth_target),
                "yield_shock_mean_target": torch.tensor(yield_shock_mean_target),
                "yield_shock_sigma_target": torch.tensor(yield_shock_sigma_target),
            }
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def load_features_df(lake_dir: Path | None = None) -> pd.DataFrame:
    """Load features.parquet from data lake."""
    path = (lake_dir or DATA_LAKE_DIR) / "features.parquet"
    return pd.read_parquet(path)


def build_dataset(seq_len: int | None = None) -> NowcastDataset:
    """Build dataset from data lake features."""
    config = get_config()
    df = load_features_df()
    sl = seq_len or config.settings.feature.sequence_length
    return NowcastDataset(df, seq_len=sl, embedding_dim=config.settings.embedding.dim)
