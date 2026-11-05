"""Shared utilities for ReefScan AI."""

from .logger import get_logger, setup_logging
from .device import resolve_device, configure_threads
from .seed import seed_everything
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
]
