# STATS 507 Final Project

**Satellite + Weather Data → Crop Nowcast → Regional Arbitrage Signals → Risk-Aware Trade Ideas → Real-Time C++ Terminal**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Terminal                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Ingest   │──▶│ Features │──▶│  Model   │──▶│ Signals  │        │
│  │ Service   │   │ Service  │   │ Service  │   │ Service  │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│       │              │              │               │               │
│       ▼              ▼              ▼               ▼               │
│  ┌─────────────────────────────────────────────────────┐           │
│  │                   Data Lake (Parquet)                │           │
│  │  regions │ weather │ imagery_index │ features        │           │
│  │  nowcasts │ macro │ trade_ideas │ prices             │           │
│  └─────────────────────────────────────────────────────┘           │
│                              │                                      │
│                              ▼                                      │
│                    ┌──────────────────┐                             │
│                    │   API Server     │                             │
│                    │  (FastAPI + WS)  │                             │
│                    │  :8777           │                             │
│                    └────────┬─────────┘                             │
│                             │ WebSocket                             │
│                             ▼                                       │
│                    ┌──────────────────┐                             │
│                    │  C++ Terminal    │                             │
│                    │  (FTXUI + IXws)  │                             │
│                    │  ┌────┬────┬───┐ │                             │
│                    │  │Macr│Reg │Trd│ │                             │
│                    │  │    │    │   │ │                             │
│                    │  └────┴────┴───┘ │                             │
│                    └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Sentinel-2/Landsat tiles ──┐
                           ├──▶ SatCLIP Encoder ──▶ E_img(r,t)
Region GeoJSON ────────────┤                       E_loc(r)
                           │                       ΔE(r,t)
Weather (ERA5/GFS-like) ───┘
         │
         ▼
  ┌─────────────────────────────┐
  │ Transformer Time-Series Head │
  │   + MC Dropout Uncertainty   │
  └──────────┬──────────────────┘
             │
             ▼
  Per-Region Nowcasts:
  • stress_index (z-scored anomaly)
  • growth_index (phenology 0–1)
  • yield_shock_mean, yield_shock_sigma
  • confidence [0,1]
  • drivers [list]
             │
             ▼
  ┌─────────────────────────────┐
  │    Signal Engine             │
  │  • Cross-region spreads      │
  │  • Substitution baskets      │
  │  • Volatility catalysts      │
  │  • Directional extremes      │
  └──────────┬──────────────────┘
             │
             ▼
  ┌─────────────────────────────┐
  │    Risk Engine               │
  │  • Pin risk (OI peaks)       │
  │  • Session/after-hours       │
  │  • Spec mismatch             │
  │  • Liquidity flags           │
  └──────────┬──────────────────┘
             │
             ▼
  Ranked Trade Ideas + Macro Indicators
  → WebSocket → C++ Terminal
```

---

## Data Sources

This project integrates data from three external sources:
- **Source**: [microsoft/SatCLIP-ViT16-L10](https://huggingface.co/microsoft/SatCLIP-ViT16-L10) on HuggingFace
- **GitHub**: [microsoft/satclip](https://github.com/microsoft/satclip)
- **Paper**: [SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery](https://arxiv.org/abs/2311.17179)

SatCLIP is a foundation model that jointly learns an **image encoder** (maps satellite tiles to embeddings) and a **location encoder** (maps lat/lon to embeddings). Trained contrastively on Sentinel-2 multi-spectral imagery so embeddings from the same location cluster together. This lets us detect **temporal changes** via ΔE and leverage **location context** without hand-crafted features.

### 2. GreenHyperSpectra — Crop Growth Imagery
- **Source**: [Avatarr05/GreenHyperSpectra](https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra) on HuggingFace
- **Paper**: GreenHyperSpectra: A multi-source hyperspectral dataset for global vegetation trait prediction

A multi-source hyperspectral dataset for global vegetation trait prediction. Provides real crop/vegetation imagery tiles used as input to the embedding pipeline. When downloaded, replaces synthetic tiles with real spectral data.

### 3. Market Prices — VEGI ETF & Ag Futures
- **Source**: [Yahoo Finance](https://finance.yahoo.com) via `yfinance`
- **VEGI**: [iShares MSCI Agriculture Producers ETF](https://www.ishares.com/us/products/239652/ishares-msci-global-agriculture-producers-etf)
- **Tickers**: VEGI, EWZ, ZC=F (Corn), ZW=F (Wheat), ZS=F (Soybeans)

Real OHLCV price data downloaded from Yahoo Finance. Used for spot price references in risk checks (pin risk), and for backtesting signals.

### Download All Data
```bash
make download-data                                    # download everything
python -m geoag.ingest.download_data --satclip        # SatCLIP weights only
python -m geoag.ingest.download_data --imagery        # GreenHyperSpectra only
python -m geoag.ingest.download_data --prices         # market prices only
python -m geoag.ingest.download_data --prices --period 1y  # 1 year of prices
```

In demo mode, the system uses a **StubBackend** for embeddings and synthetic tiles when real data hasn't been downloaded yet.

---

## How Embeddings Map to Stress/Yield Proxies

1. For each region and date, we compute `E_img(r,t)` from the satellite tile
2. We compute `ΔE(r,t) = E_img(r,t) - mean(E_img(r, t-N:t-1))` — the deviation from the trailing embedding baseline
3. We concatenate `[E_img, E_loc, ΔE, weather_features]` as input to a Transformer encoder
4. The model outputs:
   - **stress_index**: large `|ΔE|` + adverse weather → high stress
   - **growth_index**: tracks phenological stage (green-up)
   - **yield_shock_mean/sigma**: distributional estimate of yield impact
5. **MC Dropout** provides uncertainty; combined with cloud cover and data missingness to produce **confidence**

---

## How the Terminal Works

The C++ terminal:
1. Connects to `ws://localhost:8777/ws` using IXWebSocket
2. Receives JSON messages: `macro`, `regions`, `trade_ideas`, `heartbeat`
3. Parses with nlohmann/json into typed structs
4. Renders with FTXUI in three panels: Macro | Regions | Trade Ideas
5. Auto-reconnects with exponential backoff
6. Refreshes display every 1 second

### Keybinds
| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `r` | Trigger reconnect |
| `0` | All panels (default) |
| `1` | Macro panel fullscreen |
| `2` | Regions panel fullscreen |
| `3` | Trade ideas fullscreen |
| `↑/↓` | Select trade idea |
| `Enter` | Show/hide trade detail |

---


## Project Structure

```
geoag-arb-terminal/
├── README.md                          # This file
├── Makefile                           # Build + run commands
├── docker-compose.yml                 # Optional Docker deployment
├── Dockerfile.python                  # Python server container
│
├── configs/
│   ├── settings.yaml                  # Global settings
│   ├── regions.geojson                # 5 sample agricultural regions
│   ├── instruments.yaml               # Futures + ETF specifications
│   ├── calendars.yaml                 # Holidays + USDA report dates
│   ├── production_shares.yaml         # Global production weights
│   ├── export_shares.yaml             # Export share weights
│   ├── substitution_map.yaml          # Cross-commodity substitution logic
│   └── catalysts.yaml                 # Weather catalysts + OI peaks
│
├── data_samples/
│   ├── weather.csv                    # Sample weather data (ERA5-like)
│   ├── imagery/                       # GreenHyperSpectra tiles (from HuggingFace)
│   └── prices/                        # OHLCV from Yahoo Finance via yfinance
│       ├── ZC.csv                     # Corn futures (ZC=F)
│       ├── ZW.csv                     # Wheat futures (ZW=F)
│       ├── ZS.csv                     # Soybean futures (ZS=F)
│       ├── VEGI.csv                   # iShares MSCI Agriculture ETF
│       └── EWZ.csv                    # iShares MSCI Brazil ETF
│
├── python/
│   ├── pyproject.toml                 # Python package definition
│   ├── geoag/
│   │   ├── common/
│   │   │   ├── config.py              # YAML config loader + typed models
│   │   │   ├── schemas.py             # Pydantic data schemas
│   │   │   ├── timeutils.py           # Timezone + session time utilities
│   │   │   └── logging.py             # Centralized logging
│   │   ├── ingest/
│   │   │   ├── download_data.py       # HuggingFace + yfinance data downloader
│   │   │   ├── ingest_service.py      # Data ingestion orchestrator
│   │   │   └── loaders.py             # CSV/Parquet/GreenHyperSpectra loaders
│   │   ├── features/
│   │   │   ├── embedding_provider.py  # Abstract embedding interface
│   │   │   ├── stub_backend.py        # Deterministic demo embeddings
│   │   │   ├── satclip_backend.py     # Real SatCLIP (optional)
│   │   │   └── feature_service.py     # Feature pipeline
│   │   ├── model/
│   │   │   ├── model.py               # Transformer nowcast model
│   │   │   ├── dataset.py             # Sliding-window dataset
│   │   │   ├── train.py               # Training loop
│   │   │   ├── infer.py               # Inference + nowcast generation
│   │   │   └── uncertainty.py         # MC Dropout uncertainty
│   │   ├── signals/
│   │   │   ├── signal_service.py      # Signal pipeline orchestrator
│   │   │   ├── arb_logic.py           # Trade idea generation + ranking
│   │   │   ├── macro.py               # Global macro indicators
│   │   │   ├── risk.py                # Risk engine (pin, session, specs)
│   │   │   └── sessions.py            # Market session checker
│   │   └── api/
│   │       ├── server.py              # FastAPI app + lifecycle
│   │       └── ws.py                  # WebSocket connection manager
│   └── tests/
│       ├── conftest.py
│       ├── test_stub_embedding.py     # 14 embedding tests
│       ├── test_risk_engine.py        # 12 risk/session tests
│       └── test_api.py                # 5 API endpoint tests
│
├── cpp/
│   ├── CMakeLists.txt                 # CMake build (FTXUI, IXWebSocket, json)
│   └── src/
│       ├── main.cpp                   # Entry point
│       ├── ui.cpp / ui.h              # FTXUI rendering
│       ├── ws_client.cpp / ws_client.h # WebSocket client with reconnect
│       └── state.h                    # Thread-safe application state
│
└── scripts/
    ├── run_demo.sh                    # Full demo launcher
    └── download_optional_models.sh    # SatCLIP weights helper
```

---

## Configuration

All configuration is in `configs/`. Key files:

| File | Purpose |
|------|---------|
| `settings.yaml` | Global app settings, model hyperparameters, server config |
| `regions.geojson` | Agricultural regions with centroids and crop assignments |
| `instruments.yaml` | Contract specs, sessions, exchange hours |
| `calendars.yaml` | Exchange holidays + USDA report dates |
| `production_shares.yaml` | Global production weights per crop/region |
| `export_shares.yaml` | Export share weights |
| `substitution_map.yaml` | Cross-commodity substitution relationships |
| `catalysts.yaml` | Weather regime catalysts + OI peak data for pin risk |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health + version |
| GET | `/macro/latest` | Latest macro indicators |
| GET | `/regions/latest` | Latest per-region nowcasts |
| GET | `/trade_ideas/latest` | Latest ranked trade ideas |
| GET | `/instruments` | Instrument specifications |
| WS | `/ws` | Real-time streaming (macro, regions, trade_ideas, heartbeat) |

---
---

**DISCLAIMER**: This software is for educational and research purposes only. It does not constitute investment advice, and no trades are automatically executed. Past model outputs do not predict future market performance. Use at your own risk.
