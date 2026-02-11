"""Tests for the API server endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from geoag.api.server import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client (does NOT run lifespan/background tasks)."""
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "timestamp" in data
        assert data["demo_mode"] is True


class TestMacroEndpoint:
    def test_macro_latest_initial(self, client: TestClient) -> None:
        """Before pipeline runs, macro should return an error or empty."""
        resp = client.get("/macro/latest")
        assert resp.status_code == 200
        # Initially no data
        data = resp.json()
        assert "error" in data or "global_crop_stress_nowcast" in data


class TestRegionsEndpoint:
    def test_regions_latest_initial(self, client: TestClient) -> None:
        resp = client.get("/regions/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestTradeIdeasEndpoint:
    def test_trade_ideas_latest_initial(self, client: TestClient) -> None:
        resp = client.get("/trade_ideas/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestInstrumentsEndpoint:
    def test_instruments_returns_all(self, client: TestClient) -> None:
        resp = client.get("/instruments")
        assert resp.status_code == 200
        data = resp.json()
        assert "ZC" in data
        assert "ZW" in data
        assert "ZS" in data
        assert data["ZC"]["exchange"] == "CME"
