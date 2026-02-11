"""WebSocket broadcast manager for real-time updates."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket

from geoag.common.logging import get_logger
from geoag.common.timeutils import now_utc

logger = get_logger("api.ws")


class ConnectionManager:
    """Manages WebSocket connections and broadcasts updates."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        logger.info("WS client connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        logger.info("WS client disconnected (%d remaining)", len(self._connections))

    async def broadcast(self, message_type: str, data: Any) -> None:
        """Broadcast a message to all connected clients."""
        envelope = {
            "type": message_type,
            "timestamp": now_utc().isoformat(),
            "data": data,
        }
        payload = json.dumps(envelope, default=str)

        disconnected: list[WebSocket] = []
        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(payload)
                except Exception:
                    disconnected.append(ws)

            for ws in disconnected:
                self._connections.remove(ws)

        if disconnected:
            logger.info("Cleaned up %d dead connections", len(disconnected))

    async def send_heartbeat(self) -> None:
        """Send heartbeat to all clients."""
        await self.broadcast("heartbeat", {"status": "alive"})

    @property
    def connection_count(self) -> int:
        return len(self._connections)
