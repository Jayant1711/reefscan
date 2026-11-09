"""Shared utilities for ReefScan AI.

Torch-dependent helpers (device, seed, provenance) are loaded lazily so that
torch-free tooling (data cleaning, config, reporting scripts) imports without
pulling in the DL stack.
"""

from typing import Any

from .logger import get_logger, setup_logging
from .image_io import read_image_bgr, write_image_bgr, bgr_to_rgb, resize_max, ensure_uint8, list_images
from .file_utils import ensure_dir, timestamp_slug, save_json, load_json, find_latest
from .visualization import plot_training_curves, plot_confusion_matrix, image_grid

__all__ = [
    "get_logger",
    "setup_logging",
    "resolve_device",
    "configure_threads",
    "seed_everything",
    "read_image_bgr",
    "write_image_bgr",
    "bgr_to_rgb",
    "resize_max",
    "ensure_uint8",
    "list_images",
    "ensure_dir",
    "timestamp_slug",
    "save_json",
    "load_json",
    "find_latest",
    "plot_training_curves",
    "plot_confusion_matrix",
    "image_grid",
    "build_manifest",
    "count_params",
    "detect_platform",
    "git_sha",
    "gpu_name",
    "utc_now_iso",
]

_TORCH_LAZY: dict[str, tuple[str, str]] = {
    "resolve_device": (".device", "resolve_device"),
    "configure_threads": (".device", "configure_threads"),
    "seed_everything": (".seed", "seed_everything"),
    "build_manifest": (".provenance", "build_manifest"),
    "count_params": (".provenance", "count_params"),
    "detect_platform": (".provenance", "detect_platform"),
    "git_sha": (".provenance", "git_sha"),
    "gpu_name": (".provenance", "gpu_name"),
    "utc_now_iso": (".provenance", "utc_now_iso"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy loader for torch-dependent helpers."""
    if name in _TORCH_LAZY:
        from importlib import import_module

        module_name, attr = _TORCH_LAZY[name]
        return getattr(import_module(module_name, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
