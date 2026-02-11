"""Download real data from external sources.

Sources:
  1. SatCLIP weights  — HuggingFace microsoft/SatCLIP-ViT16-L10
  2. GreenHyperSpectra — HuggingFace Avatarr05/GreenHyperSpectra (crop imagery)
  3. Market prices     — yfinance (VEGI, EWZ, ZC=F, ZW=F, ZS=F)

Run:
  python -m geoag.ingest.download_data          # download everything
  python -m geoag.ingest.download_data --prices  # prices only
  python -m geoag.ingest.download_data --satclip # SatCLIP weights only
  python -m geoag.ingest.download_data --imagery # GreenHyperSpectra only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geoag.common.config import DATA_SAMPLES_DIR, REPO_ROOT
from geoag.common.logging import get_logger, setup_logging

logger = get_logger("ingest.download")

MODELS_DIR = REPO_ROOT / "models"
IMAGERY_DIR = DATA_SAMPLES_DIR / "imagery"
PRICES_DIR = DATA_SAMPLES_DIR / "prices"

# ---------------------------------------------------------------------------
# 1. SatCLIP weights from HuggingFace
# ---------------------------------------------------------------------------
SATCLIP_HF_REPO = "microsoft/SatCLIP-ViT16-L10"
SATCLIP_HF_FILE = "satclip-vit16-l10.ckpt"


def download_satclip_weights(force: bool = False) -> Path:
    """Download SatCLIP pretrained weights from HuggingFace Hub.

    Model: microsoft/SatCLIP-ViT16-L10
    Paper: https://arxiv.org/abs/2311.17179
    GitHub: https://github.com/microsoft/satclip
    """
    from huggingface_hub import hf_hub_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / SATCLIP_HF_FILE

    if target.exists() and not force:
        logger.info("SatCLIP weights already present at %s", target)
        return target

    logger.info(
        "Downloading SatCLIP weights from HuggingFace: %s/%s ...",
        SATCLIP_HF_REPO,
        SATCLIP_HF_FILE,
    )
    downloaded = hf_hub_download(
        repo_id=SATCLIP_HF_REPO,
        filename=SATCLIP_HF_FILE,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    logger.info("SatCLIP weights saved to %s", downloaded)
    return Path(downloaded)


# ---------------------------------------------------------------------------
# 2. GreenHyperSpectra crop imagery from HuggingFace
# ---------------------------------------------------------------------------
GREENHS_HF_REPO = "Avatarr05/GreenHyperSpectra"


def download_green_hyperspectra(max_samples: int = 50, force: bool = False) -> Path:
    """Download crop growth imagery from Avatarr05/GreenHyperSpectra.

    This is a multi-source hyperspectral dataset for global vegetation
    trait prediction. We download a subset of images for use as sample
    crop-growth tiles in the pipeline.

    HuggingFace: https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra
    """
    from huggingface_hub import HfApi, hf_hub_download

    IMAGERY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = IMAGERY_DIR / "greenhs_manifest.txt"

    if manifest.exists() and not force:
        logger.info("GreenHyperSpectra imagery already downloaded (manifest at %s)", manifest)
        return IMAGERY_DIR

    logger.info(
        "Downloading GreenHyperSpectra imagery from HuggingFace: %s ...",
        GREENHS_HF_REPO,
    )

    api = HfApi()
    try:
        # List files in the dataset repo
        files = api.list_repo_files(repo_id=GREENHS_HF_REPO, repo_type="dataset")

        # Filter for image files (common formats)
        image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
        image_files = [f for f in files if Path(f).suffix.lower() in image_exts]

        if not image_files:
            # If no direct image files, look for parquet/csv data files
            data_files = [f for f in files if Path(f).suffix.lower() in {".parquet", ".csv", ".arrow"}]
            if data_files:
                logger.info("Found %d data files in GreenHyperSpectra", len(data_files))
                # Download first few data files
                downloaded_paths = []
                for df in data_files[:max_samples]:
                    try:
                        p = hf_hub_download(
                            repo_id=GREENHS_HF_REPO,
                            filename=df,
                            repo_type="dataset",
                            local_dir=str(IMAGERY_DIR / "greenhs"),
                            local_dir_use_symlinks=False,
                        )
                        downloaded_paths.append(p)
                    except Exception as exc:
                        logger.warning("Failed to download %s: %s", df, exc)

                manifest.write_text("\n".join(downloaded_paths))
                logger.info("Downloaded %d GreenHyperSpectra data files", len(downloaded_paths))
                return IMAGERY_DIR

            # Also try downloading the README at minimum to confirm access
            try:
                readme_path = hf_hub_download(
                    repo_id=GREENHS_HF_REPO,
                    filename="README.md",
                    repo_type="dataset",
                    local_dir=str(IMAGERY_DIR / "greenhs"),
                    local_dir_use_symlinks=False,
                )
                logger.info("Downloaded GreenHyperSpectra README: %s", readme_path)
            except Exception:
                pass

            # Download whatever is available
            all_downloadable = [f for f in files if not f.startswith(".")]
            downloaded_paths = []
            for af in all_downloadable[:max_samples]:
                try:
                    p = hf_hub_download(
                        repo_id=GREENHS_HF_REPO,
                        filename=af,
                        repo_type="dataset",
                        local_dir=str(IMAGERY_DIR / "greenhs"),
                        local_dir_use_symlinks=False,
                    )
                    downloaded_paths.append(p)
                except Exception as exc:
                    logger.debug("Skipped %s: %s", af, exc)

            manifest.write_text("\n".join(downloaded_paths) if downloaded_paths else "no_files")
            logger.info(
                "Downloaded %d files from GreenHyperSpectra (repo has %d files total)",
                len(downloaded_paths),
                len(files),
            )
        else:
            # Download image files directly
            downloaded_paths = []
            for img in image_files[:max_samples]:
                try:
                    p = hf_hub_download(
                        repo_id=GREENHS_HF_REPO,
                        filename=img,
                        repo_type="dataset",
                        local_dir=str(IMAGERY_DIR / "greenhs"),
                        local_dir_use_symlinks=False,
                    )
                    downloaded_paths.append(p)
                except Exception as exc:
                    logger.warning("Failed to download %s: %s", img, exc)

            manifest.write_text("\n".join(downloaded_paths))
            logger.info("Downloaded %d GreenHyperSpectra images", len(downloaded_paths))

    except Exception as exc:
        logger.warning(
            "Could not access GreenHyperSpectra repo (%s). "
            "The pipeline will fall back to synthetic tiles. Error: %s",
            GREENHS_HF_REPO,
            exc,
        )
        manifest.write_text("error: " + str(exc))

    return IMAGERY_DIR


# ---------------------------------------------------------------------------
# 3. Market prices via yfinance
# ---------------------------------------------------------------------------
# Yahoo Finance ticker mapping:
#   ZC=F  → Corn futures
#   ZW=F  → Wheat futures
#   ZS=F  → Soybean futures
#   VEGI  → iShares MSCI Agriculture Producers ETF
#   EWZ   → iShares MSCI Brazil ETF
PRICE_TICKERS = {
    "ZC": "ZC=F",
    "ZW": "ZW=F",
    "ZS": "ZS=F",
    "VEGI": "VEGI",
    "EWZ": "EWZ",
}


def download_prices(
    period: str = "6mo",
    force: bool = False,
) -> dict[str, Path]:
    """Download historical OHLCV prices from Yahoo Finance via yfinance.

    Source: https://finance.yahoo.com
    ETF info: https://www.ishares.com/us/products/239652/ishares-msci-global-agriculture-producers-etf

    Args:
        period: yfinance period string (e.g., "6mo", "1y", "2y").
        force: Re-download even if CSVs already exist.

    Returns:
        Dict mapping our symbol → saved CSV path.
    """
    import yfinance as yf

    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for our_sym, yf_ticker in PRICE_TICKERS.items():
        csv_path = PRICES_DIR / f"{our_sym}.csv"

        if csv_path.exists() and not force:
            logger.info("Price data for %s already exists at %s", our_sym, csv_path)
            results[our_sym] = csv_path
            continue

        logger.info("Downloading %s (%s) from Yahoo Finance ...", our_sym, yf_ticker)
        try:
            ticker = yf.Ticker(yf_ticker)
            df = ticker.history(period=period)

            if df.empty:
                logger.warning("No data returned for %s — keeping existing CSV if any", yf_ticker)
                if csv_path.exists():
                    results[our_sym] = csv_path
                continue

            # Normalize columns to match our format: date,open,high,low,close,volume
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Keep only the columns we need
            keep_cols = ["date", "open", "high", "low", "close", "volume"]
            available = [c for c in keep_cols if c in df.columns]
            df = df[available]

            # Format date
            if "date" in df.columns:
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")

            # Round prices
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].round(2)
            if "volume" in df.columns:
                df["volume"] = df["volume"].astype(int)

            df.to_csv(csv_path, index=False)
            logger.info("Saved %d rows for %s to %s", len(df), our_sym, csv_path)
            results[our_sym] = csv_path

        except Exception as exc:
            logger.warning("Failed to download %s: %s — keeping existing CSV if any", yf_ticker, exc)
            if csv_path.exists():
                results[our_sym] = csv_path

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def download_all(force: bool = False) -> None:
    """Download all external data sources."""
    logger.info("=== GeoAg Arb Terminal — Data Download ===")

    logger.info("\n--- 1/3: SatCLIP Weights ---")
    try:
        download_satclip_weights(force=force)
    except Exception as exc:
        logger.warning("SatCLIP download failed: %s (will use StubBackend)", exc)

    logger.info("\n--- 2/3: GreenHyperSpectra Imagery ---")
    try:
        download_green_hyperspectra(force=force)
    except Exception as exc:
        logger.warning("GreenHyperSpectra download failed: %s (will use synthetic tiles)", exc)

    logger.info("\n--- 3/3: Market Prices ---")
    try:
        download_prices(force=force)
    except Exception as exc:
        logger.warning("Price download failed: %s (will use existing CSVs)", exc)

    logger.info("\n=== Download complete ===")


def main() -> None:
    setup_logging("INFO")

    parser = argparse.ArgumentParser(description="Download data for GeoAg Arb Terminal")
    parser.add_argument("--satclip", action="store_true", help="Download SatCLIP weights only")
    parser.add_argument("--imagery", action="store_true", help="Download GreenHyperSpectra only")
    parser.add_argument("--prices", action="store_true", help="Download market prices only")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--period", default="6mo", help="Price history period (default: 6mo)")
    args = parser.parse_args()

    specific = args.satclip or args.imagery or args.prices

    if not specific:
        download_all(force=args.force)
    else:
        if args.satclip:
            download_satclip_weights(force=args.force)
        if args.imagery:
            download_green_hyperspectra(force=args.force)
        if args.prices:
            download_prices(period=args.period, force=args.force)


if __name__ == "__main__":
    main()
