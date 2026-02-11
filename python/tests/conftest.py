"""Shared test fixtures."""

from __future__ import annotations

import pytest

from geoag.common.config import CONFIGS_DIR, ConfigStore, reset_config


@pytest.fixture(autouse=True)
def _reset_global_config() -> None:
    """Reset global config between tests."""
    reset_config()


@pytest.fixture
def config() -> ConfigStore:
    """Provide a fresh ConfigStore."""
    return ConfigStore(CONFIGS_DIR)
