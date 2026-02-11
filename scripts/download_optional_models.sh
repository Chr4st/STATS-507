#!/usr/bin/env bash
# Download SatCLIP model weights + GreenHyperSpectra imagery + market prices
# ===========================================================================
#
# Data sources:
#   1. SatCLIP weights:     HuggingFace microsoft/SatCLIP-ViT16-L10
#      GitHub:              https://github.com/microsoft/satclip
#   2. GreenHyperSpectra:   HuggingFace Avatarr05/GreenHyperSpectra
#      (crop growth hyperspectral imagery for vegetation trait prediction)
#   3. Market prices:       Yahoo Finance via yfinance
#      VEGI ETF:            https://www.ishares.com/us/products/239652/
#
# Usage:
#   ./scripts/download_optional_models.sh           # download everything
#   ./scripts/download_optional_models.sh --satclip  # SatCLIP only
#   ./scripts/download_optional_models.sh --prices   # prices only
#   ./scripts/download_optional_models.sh --imagery  # GreenHyperSpectra only
#
# Or use the Python module directly:
#   python -m geoag.ingest.download_data
#   python -m geoag.ingest.download_data --satclip
#   python -m geoag.ingest.download_data --prices --period 1y

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== GeoAg Arb Terminal — Data Download ==="
echo ""
echo "Sources:"
echo "  - SatCLIP weights:   microsoft/SatCLIP-ViT16-L10 (HuggingFace)"
echo "  - Crop imagery:      Avatarr05/GreenHyperSpectra (HuggingFace)"
echo "  - Market prices:     VEGI, EWZ, ZC, ZW, ZS (Yahoo Finance)"
echo ""

# Activate venv if it exists
if [ -f "$REPO_ROOT/python/.venv/bin/activate" ]; then
    source "$REPO_ROOT/python/.venv/bin/activate"
fi

cd "$REPO_ROOT/python"

# Pass through all arguments to the Python downloader
python -m geoag.ingest.download_data "$@"

echo ""
echo "Done. Check data_samples/ and models/ for downloaded files."
