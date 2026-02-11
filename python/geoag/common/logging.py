"""Centralized logging configuration for GeoAg Arb Terminal."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", fmt: str | None = None) -> None:
    """Configure root logger with consistent format."""
    log_format = fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(f"geoag.{name}")
