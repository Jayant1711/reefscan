"""Device selection and CPU thread configuration (Colab / Kaggle / laptop friendly)."""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def resolve_device(prefer_gpu: bool = True) -> torch.device:
    """Pick the best available torch device."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif prefer_gpu and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
        logger.info("Using Apple MPS backend.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (%d logical cores).", torch.cpu.thread_count() if hasattr(torch.cpu, "thread_count") else -1)
    return device


def configure_threads(num_threads: int = 0) -> None:
    """Optionally cap intra/inter-op threads (0 = use defaults)."""
    if num_threads and num_threads > 0:
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialised
        logger.info("Torch threads capped at %d.", num_threads)


def device_summary(device: torch.device) -> dict[str, Any]:
    """Human-readable device info for logs and reports."""
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info
