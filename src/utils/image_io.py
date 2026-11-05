"""Image I/O helpers built on OpenCV (BGR convention throughout the codebase)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. Raises FileNotFoundError / ValueError on failure."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    # IMREAD_COLOR: 3-channel BGR; ignore EXIF rotation issues by using unchanged+convert
    data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_image_bgr(image: np.ndarray, path: str | Path) -> Path:
    """Write a BGR uint8 image (unicode-safe on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"Failed to encode image for {path}")
    buf.tofile(str(path))
    return path


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is <= max_side, keeping aspect ratio."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Clip/convert any float image to uint8."""
    if image.dtype == np.uint8:
        return image
    arr = np.clip(image, 0, 255)
    return arr.astype(np.uint8)


def list_images(directory: str | Path, recursive: bool = True) -> list[Path]:
    """List image files under a directory."""
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in directory.glob(pattern) if p.suffix.lower() in SUPPORTED_IMAGE_EXTS and p.is_file())
