"""Logging configuration: console + rotating file output."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger once for the whole application."""
    global _configured
    if _configured:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        )

    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (setup_logging must have been called for file output)."""
    return logging.getLogger(name)
