"""Filesystem helpers: dirs, timestamps, JSON, artifact discovery."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
    logger.debug("Saved JSON -> %s", path)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def find_latest(directory: str | Path, pattern: str = "*") -> Path | None:
    """Newest file matching a glob pattern, or None."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob(pattern) if p.is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime, default=None)


def sanitize_name(name: str) -> str:
    """Filesystem-safe name fragment."""
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "item"
