"""Ingest service: loads raw data, writes to data lake as Parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from geoag.common.config import DATA_LAKE_DIR, ConfigStore, get_config
from geoag.common.logging import get_logger
from geoag.ingest.loaders import (
    build_imagery_index,
    load_all_prices,
    load_weather,
)

logger = get_logger("ingest.service")


class IngestService:
    """Orchestrates data ingestion into the data lake."""

    def __init__(self, config: ConfigStore | None = None) -> None:
        self.config = config or get_config()
        self.lake_dir = DATA_LAKE_DIR
        self.lake_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Path]:
        """Execute full ingestion pipeline. Returns paths to written files."""
        outputs: dict[str, Path] = {}

        # 1. Regions
        regions_df = pd.DataFrame([r.model_dump() for r in self.config.regions])
        regions_path = self.lake_dir / "regions.parquet"
        regions_df.to_parquet(regions_path, index=False)
        outputs["regions"] = regions_path
        logger.info("Wrote %d regions to %s", len(regions_df), regions_path)

        # 2. Weather
        weather_df = load_weather()
        weather_path = self.lake_dir / "weather.parquet"
        weather_df.to_parquet(weather_path, index=False)
        outputs["weather"] = weather_path
        logger.info("Wrote %d weather rows to %s", len(weather_df), weather_path)

        # 3. Imagery index (synthetic)
        dates = sorted(weather_df["date"].dt.strftime("%Y-%m-%d").unique().tolist())
        imagery_df = build_imagery_index(
            self.config.regions, dates, seed=self.config.settings.app.seed
        )
        imagery_path = self.lake_dir / "imagery_index.parquet"
        imagery_df.to_parquet(imagery_path, index=False)
        outputs["imagery_index"] = imagery_path
        logger.info("Wrote %d imagery index rows to %s", len(imagery_df), imagery_path)

        # 4. Prices
        symbols = list(self.config.instruments.keys())
        prices = load_all_prices(symbols)
        for sym, df in prices.items():
            price_path = self.lake_dir / f"prices_{sym}.parquet"
            df.to_parquet(price_path, index=False)
            outputs[f"prices_{sym}"] = price_path
        logger.info("Wrote prices for %d instruments", len(prices))

        return outputs


def run_ingest() -> dict[str, Path]:
    """Convenience entry point."""
    svc = IngestService()
    return svc.run()


if __name__ == "__main__":
    from geoag.common.logging import setup_logging

    setup_logging("INFO")
    run_ingest()
