"""FastAPI server: REST + WebSocket endpoints for the GeoAg Arb Terminal."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from geoag.api.ws import ConnectionManager
from geoag.common.config import get_config
from geoag.common.logging import get_logger, setup_logging
from geoag.common.schemas import HealthResponse
from geoag.common.timeutils import now_utc
from geoag.features.feature_service import FeatureService
from geoag.ingest.ingest_service import IngestService
from geoag.model.train import train_model
from geoag.signals.signal_service import SignalService

logger = get_logger("api.server")

# Global state
manager = ConnectionManager()
_latest_state: dict[str, object] = {
    "nowcasts": [],
    "macro": None,
    "trade_ideas": [],
}
_shutdown_event = asyncio.Event()


async def pipeline_loop(interval: int) -> None:
    """Periodically re-run signal pipeline and broadcast updates."""
    config = get_config()

    while not _shutdown_event.is_set():
        try:
            logger.info("Running signal pipeline...")
            svc = SignalService(config)
            result = svc.run()

            _latest_state["nowcasts"] = result["nowcasts"]
            _latest_state["macro"] = result["macro"]
            _latest_state["trade_ideas"] = result["trade_ideas"]

            # Broadcast to WS clients
            nowcasts = result["nowcasts"]
            macro = result["macro"]
            ideas = result["trade_ideas"]

            if macro:
                await manager.broadcast("macro", macro.model_dump() if hasattr(macro, "model_dump") else {})

            if nowcasts:
                await manager.broadcast(
                    "regions",
                    [nc.model_dump() for nc in nowcasts] if nowcasts else [],
                )

            if ideas:
                await manager.broadcast(
                    "trade_ideas",
                    [ti.model_dump() for ti in ideas] if ideas else [],
                )

            logger.info(
                "Pipeline complete: %d nowcasts, %d ideas, %d WS clients",
                len(nowcasts) if nowcasts else 0,
                len(ideas) if ideas else 0,
                manager.connection_count,
            )

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)

        # Wait for interval or shutdown
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=interval)
            break  # Shutdown requested
        except TimeoutError:
            pass  # Normal timeout, loop again


async def heartbeat_loop(interval: int) -> None:
    """Send periodic heartbeat to WS clients."""
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=interval)
            break
        except TimeoutError:
            await manager.send_heartbeat()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup/shutdown lifecycle."""
    config = get_config()
    setup_logging(config.settings.app.name)

    logger.info("Starting GeoAg Arb Terminal API v%s", config.settings.app.version)

    # Run initial data ingestion
    logger.info("Running initial data ingestion...")
    ingest_svc = IngestService(config)
    ingest_svc.run()

    # Run feature pipeline
    logger.info("Running feature pipeline...")
    feature_svc = FeatureService(config)
    feature_svc.run()

    # Train model (quick demo training)
    logger.info("Training model (demo)...")
    train_model(epochs=20, lr=1e-3, batch_size=4)

    # Start background tasks
    refresh_interval = config.settings.feature.refresh_interval_s
    heartbeat_interval = config.settings.server.heartbeat_interval_s

    pipeline_task = asyncio.create_task(pipeline_loop(refresh_interval))
    heartbeat_task = asyncio.create_task(heartbeat_loop(heartbeat_interval))

    yield

    # Shutdown
    logger.info("Shutting down...")
    _shutdown_event.set()
    pipeline_task.cancel()
    heartbeat_task.cancel()
    with suppress(asyncio.CancelledError):
        await pipeline_task
    with suppress(asyncio.CancelledError):
        await heartbeat_task


app = FastAPI(
    title="GeoAg Arb Terminal API",
    version="0.1.0",
    description="Satellite + Weather → Crop Nowcast → Regional Arb Signals",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    config = get_config()
    return HealthResponse(
        status="ok",
        version=config.settings.app.version,
        timestamp=now_utc(),
        demo_mode=config.settings.app.demo_mode,
    )


@app.get("/macro/latest")
async def macro_latest() -> dict:
    macro = _latest_state.get("macro")
    if macro is None:
        return {"error": "No macro data available yet"}
    if hasattr(macro, "model_dump"):
        return macro.model_dump()
    return {}


@app.get("/regions/latest")
async def regions_latest() -> list[dict]:
    nowcasts = _latest_state.get("nowcasts", [])
    return [nc.model_dump() for nc in nowcasts] if nowcasts else []


@app.get("/trade_ideas/latest")
async def trade_ideas_latest() -> list[dict]:
    ideas = _latest_state.get("trade_ideas", [])
    return [ti.model_dump() for ti in ideas] if ideas else []


@app.get("/instruments")
async def instruments() -> dict:
    config = get_config()
    return {
        sym: spec.model_dump() for sym, spec in config.instruments.items()
    }


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        # Send current state on connect
        macro = _latest_state.get("macro")
        if macro and hasattr(macro, "model_dump"):
            await ws.send_json({
                "type": "macro",
                "timestamp": now_utc().isoformat(),
                "data": macro.model_dump(),
            })

        nowcasts = _latest_state.get("nowcasts", [])
        if nowcasts:
            await ws.send_json({
                "type": "regions",
                "timestamp": now_utc().isoformat(),
                "data": [nc.model_dump() for nc in nowcasts],
            })

        ideas = _latest_state.get("trade_ideas", [])
        if ideas:
            await ws.send_json({
                "type": "trade_ideas",
                "timestamp": now_utc().isoformat(),
                "data": [ti.model_dump() for ti in ideas],
            })

        # Keep alive — listen for client messages (e.g., ping)
        while True:
            data = await ws.receive_text()
            # Echo pings
            if data == "ping":
                await ws.send_text('{"type":"pong"}')

    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception as exc:
        logger.warning("WS error: %s", exc)
        await manager.disconnect(ws)


def main() -> None:
    """Run the server."""
    import uvicorn

    setup_logging("INFO")
    config = get_config()
    uvicorn.run(
        "geoag.api.server:app",
        host=config.settings.server.host,
        port=config.settings.server.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
