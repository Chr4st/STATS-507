# GeoAg Arb Terminal — Makefile
# ============================================
.PHONY: setup setup-python setup-cpp download-data demo demo-server demo-terminal \
        test test-python lint typecheck clean help

PYTHON_DIR := python
CPP_DIR := cpp
CPP_BUILD_DIR := $(CPP_DIR)/build

PYTHON_DIR_ABS := $(shell cd $(PYTHON_DIR) && pwd)
VENV := $(PYTHON_DIR)/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UV := uv

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "GeoAg Arb Terminal"
	@echo "===================="
	@echo ""
	@echo "  make setup          Install all dependencies (Python + C++)"
	@echo "  make setup-python   Install Python dependencies only"
	@echo "  make setup-cpp      Build C++ terminal only"
	@echo "  make download-data  Download real data (SatCLIP, GreenHyperSpectra, prices)"
	@echo "  make demo           Start everything (server + terminal)"
	@echo "  make demo-server    Start Python API server only"
	@echo "  make demo-terminal  Start C++ terminal only"
	@echo "  make test           Run all tests"
	@echo "  make lint           Run ruff linter"
	@echo "  make typecheck      Run mypy type checker"
	@echo "  make clean          Remove build artifacts"
	@echo ""
	@echo "DISCLAIMER: For research only; not investment advice."

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup: setup-python setup-cpp
	@echo "✓ Setup complete"

setup-python:
	@echo "→ Creating Python virtual environment..."
	cd $(PYTHON_DIR) && $(UV) venv --python 3.11 .venv 2>/dev/null || cd $(PYTHON_DIR) && python3 -m venv .venv
	@echo "→ Installing Python dependencies..."
	cd $(PYTHON_DIR) && $(UV) pip install --python .venv/bin/python -e ".[dev]"
	@echo "✓ Python setup complete"

download-data:
	@echo "→ Downloading real data from HuggingFace + Yahoo Finance..."
	@echo "  Sources: microsoft/SatCLIP-ViT16-L10, Avatarr05/GreenHyperSpectra, yfinance"
	cd $(PYTHON_DIR) && source .venv/bin/activate && $(PYTHON) -m geoag.ingest.download_data
	@echo "✓ Data download complete"

setup-cpp:
	@echo "→ Building C++ terminal..."
	cmake -S $(CPP_DIR) -B $(CPP_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CPP_BUILD_DIR) --parallel
	@echo "✓ C++ terminal built at $(CPP_BUILD_DIR)/terminal"

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
demo:
	@echo "Starting GeoAg Arb Terminal Demo..."
	@echo "DISCLAIMER: For research only; not investment advice."
	@echo ""
	@echo "Starting Python API server in background..."
	cd $(PYTHON_DIR) && source .venv/bin/activate && $(PYTHON) -m geoag.api.server &
	@sleep 10
	@echo "Starting C++ terminal..."
	$(CPP_BUILD_DIR)/terminal --url ws://localhost:8777/ws

demo-server:
	@echo "Starting Python API server..."
	@echo "DISCLAIMER: For research only; not investment advice."
	cd $(PYTHON_DIR) && source .venv/bin/activate && $(PYTHON) -m geoag.api.server

demo-terminal:
	@echo "Starting C++ terminal..."
	@echo "DISCLAIMER: For research only; not investment advice."
	$(CPP_BUILD_DIR)/terminal --url ws://localhost:8777/ws

# ---------------------------------------------------------------------------
# Testing & Quality
# ---------------------------------------------------------------------------
test: test-python
	@echo "✓ All tests passed"

test-python:
	@echo "→ Running Python tests..."
	cd $(PYTHON_DIR) && source .venv/bin/activate && $(PYTHON) -m pytest tests/ -v --tb=short

lint:
	@echo "→ Running ruff..."
	cd $(PYTHON_DIR) && source .venv/bin/activate && $(PYTHON) -m ruff check geoag/ tests/

typecheck:
	@echo "→ Running mypy..."
	cd $(PYTHON_DIR) && source .venv/bin/activate && $(PYTHON) -m mypy geoag/ --ignore-missing-imports

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -rf $(CPP_BUILD_DIR)
	rm -rf data_lake/
	rm -rf $(PYTHON_DIR)/__pycache__
	find $(PYTHON_DIR) -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find $(PYTHON_DIR) -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"
