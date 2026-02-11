"""Feature service: produces embeddings and derived features for each region/date."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from geoag.common.config import DATA_LAKE_DIR, ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.features.embedding_provider import EmbeddingProvider
from geoag.features.satclip_backend import SatCLIPBackend
from geoag.features.stub_backend import StubBackend
from geoag.ingest.loaders import get_tile_for_region

logger = get_logger("features.service")


def create_embedding_provider(config: ConfigStore) -> EmbeddingProvider:
    """Factory: create the appropriate embedding backend."""
    backend = config.settings.embedding.backend
    dim = config.settings.embedding.dim
    if backend == "satclip":
        return SatCLIPBackend(
            weights_path=config.settings.embedding.satclip_weights,
            dim=dim,
        )
    return StubBackend(dim=dim, seed=config.settings.app.seed)


class FeatureService:
    """Computes embeddings and derived features, writes features.parquet."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()
        self.provider = create_embedding_provider(self.config)
        self.lake_dir = DATA_LAKE_DIR
        self.dim = self.provider.embedding_dim

    def run(self) -> Path:
        """Compute features for all region/date combinations in the imagery index."""
        imagery_path = self.lake_dir / "imagery_index.parquet"
        weather_path = self.lake_dir / "weather.parquet"

        imagery_df = pd.read_parquet(imagery_path)
        weather_df = pd.read_parquet(weather_path)

        records: list[dict] = []
        region_map = self.config.region_map

        # Group by region, process chronologically
        for region_id, group in imagery_df.groupby("region_id"):
            region = region_map.get(str(region_id))
            if region is None:
                continue

            # Location embedding (static per region)
            e_loc = self.provider.encode_location(region.centroid_lat, region.centroid_lon)

            # Accumulate image embeddings for trailing mean
            trailing_embeddings: list[torch.Tensor] = []

            for _, row in group.sort_values("date").iterrows():
                date_str = str(row["date"])

                # Get tile (GreenHyperSpectra if available, else synthetic)
                tile = get_tile_for_region(
                    region, date_str, seed=self.config.settings.app.seed
                )

                # Image embedding
                e_img = self.provider.encode_image(tile)
                trailing_embeddings.append(e_img)

                # Delta vs trailing mean
                window = self.config.settings.feature.seasonal_window
                if len(trailing_embeddings) > 1:
                    trail = torch.stack(trailing_embeddings[-window:])
                    trail_mean = trail.mean(dim=0)
                    delta_e = e_img - trail_mean
                else:
                    delta_e = torch.zeros(self.dim)

                # Weather features for this region/date
                w_row = weather_df[
                    (weather_df["region_id"] == region_id)
                    & (weather_df["date"].astype(str) == date_str)
                ]

                weather_feats = self._extract_weather_features(w_row, region)
                cloud_cover = float(row.get("cloud_cover", 0.0))

                record = {
                    "date": date_str,
                    "region_id": str(region_id),
                    "cloud_cover": cloud_cover,
                }

                # Store embeddings as lists (Parquet-friendly)
                for i in range(self.dim):
                    record[f"e_img_{i}"] = float(e_img[i])
                    record[f"e_loc_{i}"] = float(e_loc[i])
                    record[f"delta_e_{i}"] = float(delta_e[i])

                record.update(weather_feats)
                records.append(record)

        features_df = pd.DataFrame(records)
        out_path = self.lake_dir / "features.parquet"
        features_df.to_parquet(out_path, index=False)
        logger.info("Wrote %d feature rows to %s", len(features_df), out_path)
        return out_path

    def compute_latest(self) -> pd.DataFrame:
        """Compute features for the most recent date only (for real-time inference)."""
        features_path = self.lake_dir / "features.parquet"
        if features_path.exists():
            df = pd.read_parquet(features_path)
            latest_date = df["date"].max()
            return df[df["date"] == latest_date]
        # If no features yet, run full pipeline
        self.run()
        df = pd.read_parquet(features_path)
        latest_date = df["date"].max()
        return df[df["date"] == latest_date]

    def _extract_weather_features(
        self, w_row: pd.DataFrame, region: object
    ) -> dict[str, float]:
        """Extract weather anomaly features."""
        if w_row.empty:
            return {
                "gdd": 0.0,
                "rainfall_anomaly": 0.0,
                "heat_stress_days": 0.0,
                "drought_proxy": 0.0,
                "temp_max": 0.0,
                "temp_min": 0.0,
                "precip_mm": 0.0,
                "soil_moisture": 0.0,
            }

        row = w_row.iloc[0]
        temp_max = float(row.get("temp_max_c", 0))
        temp_min = float(row.get("temp_min_c", 0))
        precip = float(row.get("precip_mm", 0))
        soil = float(row.get("soil_moisture_pct", 0))

        # Growing Degree Days (base 10°C)
        gdd = max(0.0, (temp_max + temp_min) / 2.0 - 10.0)

        # Heat stress: days above 35°C threshold
        heat_stress = 1.0 if temp_max > 35.0 else 0.0

        # Rainfall anomaly (vs 5mm/day rough baseline)
        rainfall_anomaly = precip - 5.0

        # Drought proxy (inverted soil moisture, normalized)
        drought_proxy = max(0.0, (50.0 - soil) / 50.0)

        return {
            "gdd": round(gdd, 2),
            "rainfall_anomaly": round(rainfall_anomaly, 2),
            "heat_stress_days": heat_stress,
            "drought_proxy": round(drought_proxy, 3),
            "temp_max": temp_max,
            "temp_min": temp_min,
            "precip_mm": precip,
            "soil_moisture": soil,
        }


def run_features() -> Path:
    """Convenience entry point."""
    svc = FeatureService()
    return svc.run()


if __name__ == "__main__":
    from geoag.common.logging import setup_logging

    setup_logging("INFO")
    run_features()
