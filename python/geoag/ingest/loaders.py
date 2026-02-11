"""Data loaders for regions, weather, imagery, and prices.

Supports both real downloaded data and synthetic fallbacks:
- Prices: CSV files from yfinance (Yahoo Finance)
- Imagery: GreenHyperSpectra from HuggingFace (Avatarr05/GreenHyperSpectra),
           falls back to synthetic tiles if not available
- Weather: CSV sample data (ERA5-like format)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from geoag.common.config import DATA_SAMPLES_DIR, RegionInfo
from geoag.common.logging import get_logger

logger = get_logger("ingest.loaders")

IMAGERY_DIR = DATA_SAMPLES_DIR / "imagery"


def load_weather(data_dir: Path | None = None) -> pd.DataFrame:
    """Load weather CSV into DataFrame."""
    path = (data_dir or DATA_SAMPLES_DIR) / "weather.csv"
    logger.info("Loading weather from %s", path)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_prices(symbol: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Load OHLCV CSV for a given instrument symbol.

    Data sourced from Yahoo Finance via yfinance.
    Run `python -m geoag.ingest.download_data --prices` to fetch latest.
    """
    path = (data_dir or DATA_SAMPLES_DIR) / "prices" / f"{symbol}.csv"
    if not path.exists():
        logger.warning(
            "Price CSV not found for %s at %s. "
            "Run: python -m geoag.ingest.download_data --prices",
            symbol,
            path,
        )
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    logger.info("Loading prices for %s from %s", symbol, path)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_all_prices(symbols: list[str], data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load prices for multiple symbols. Skips missing files gracefully."""
    results: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_prices(sym, data_dir)
        if not df.empty:
            results[sym] = df
    return results


def _find_greenhs_tiles() -> list[Path]:
    """Find downloaded GreenHyperSpectra image tiles."""
    greenhs_dir = IMAGERY_DIR / "greenhs"
    if not greenhs_dir.exists():
        return []
    tiles: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.npy"):
        tiles.extend(greenhs_dir.rglob(ext))
    return sorted(tiles)


def load_greenhs_tile(path: Path) -> np.ndarray:
    """Load a GreenHyperSpectra tile as a numpy array.

    Source: HuggingFace Avatarr05/GreenHyperSpectra
    Paper: GreenHyperSpectra — multi-source hyperspectral dataset
           for global vegetation trait prediction
    """
    if path.suffix == ".npy":
        arr = np.load(path)
        # If hyperspectral (H, W, >3 bands), take first 3 as RGB proxy
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            # Normalize to uint8
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return arr

    # PIL-based loading for standard image formats
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        return np.array(img)
    except ImportError:
        # Fallback: just generate synthetic
        logger.debug("PIL not available, using synthetic tile for %s", path)
        return generate_synthetic_tile(
            RegionInfo(
                id="fallback", name="fallback", crop="unknown",
                country="XX", centroid_lat=0, centroid_lon=0,
            ),
            "2026-01-01",
        )


def get_tile_for_region(
    region: RegionInfo,
    date_str: str,
    seed: int = 42,
    size: int = 64,
) -> np.ndarray:
    """Get an imagery tile for a region/date.

    Tries GreenHyperSpectra tiles first (round-robin mapped by region+date),
    falls back to synthetic generation.
    """
    greenhs_tiles = _find_greenhs_tiles()

    if greenhs_tiles:
        # Deterministic mapping: hash region+date → tile index
        hash_input = f"{region.id}_{date_str}_{seed}"
        h = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        idx = h % len(greenhs_tiles)
        try:
            tile = load_greenhs_tile(greenhs_tiles[idx])
            # Resize if needed
            if tile.shape[0] != size or tile.shape[1] != size:
                try:
                    from PIL import Image

                    img = Image.fromarray(tile).resize((size, size))
                    tile = np.array(img)
                except ImportError:
                    # Crop/pad manually
                    tile = tile[:size, :size]
                    if tile.shape[0] < size or tile.shape[1] < size:
                        padded = np.zeros((size, size, 3), dtype=np.uint8)
                        padded[: tile.shape[0], : tile.shape[1]] = tile
                        tile = padded
            return tile
        except Exception as exc:
            logger.debug("Failed to load GreenHS tile %s: %s", greenhs_tiles[idx], exc)

    # Fallback to synthetic
    return generate_synthetic_tile(region, date_str, seed, size)


def generate_synthetic_tile(
    region: RegionInfo,
    date_str: str,
    seed: int = 42,
    size: int = 64,
) -> np.ndarray:
    """Generate a deterministic synthetic RGB tile for a region/date.

    Uses region centroid + date as seed for reproducibility.
    Returns (size, size, 3) uint8 array simulating an NDVI-like image.
    """
    hash_input = f"{region.id}_{date_str}_{seed}"
    h = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(h)

    # Base "greenness" varies by latitude (higher lat = less green in winter)
    lat_factor = max(0.2, 1.0 - abs(region.centroid_lat) / 90.0)

    green = rng.randint(40, int(200 * lat_factor + 56), size=(size, size), dtype=np.uint8)
    red = rng.randint(20, 100, size=(size, size), dtype=np.uint8)
    blue = rng.randint(10, 80, size=(size, size), dtype=np.uint8)

    tile = np.stack([red, green, blue], axis=-1)
    return tile


def build_imagery_index(
    regions: list[RegionInfo],
    dates: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Build an imagery index DataFrame.

    When GreenHyperSpectra data is available, source is tagged as 'greenhs'.
    Otherwise defaults to 'synthetic'.
    """
    greenhs_tiles = _find_greenhs_tiles()
    source = "greenhs" if greenhs_tiles else "synthetic"
    if greenhs_tiles:
        logger.info(
            "Using %d GreenHyperSpectra tiles (Avatarr05/GreenHyperSpectra) as imagery source",
            len(greenhs_tiles),
        )

    records = []
    for region in regions:
        for date_str in dates:
            hash_input = f"{region.id}_{date_str}_{seed}"
            tile_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
            cloud_cover = max(0.0, min(1.0, abs(hash(hash_input) % 100) / 100.0 * 0.4))
            records.append(
                {
                    "date": date_str,
                    "region_id": region.id,
                    "tile_hash": tile_hash,
                    "cloud_cover": round(cloud_cover, 3),
                    "source": source,
                    "bands": "RGB",
                    "resolution_m": 10,
                }
            )
    return pd.DataFrame(records)
