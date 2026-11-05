"""White balance correction: gray-world + shades-of-gray (Minkowski p-norm)."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register


@register
class WhiteBalanceEnhancer(Enhancer):
    """Shades-of-gray colour constancy.

    Water absorbs red fastest, so underwater images have a cast; scaling each
    channel toward a common Minkowski-norm brightness removes most of it.
    A percentile clip guards against extreme statistics.
    """

    name = "white_balance"
    description = "Shades-of-gray white balance (Minkowski p=6) with percentile clipping"
    reference = "Bianco et al., 2017; Buchsbaum, 1980 (gray world)"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        p = 6.0
        percentile = float(self.params.get("wb_percentile", 2.0))

        img = image.astype(np.float32)
        # Percentile clip to suppress outliers (bright fish flashes, sun glints).
        lo = np.percentile(img, percentile, axis=(0, 1), keepdims=True)
        hi = np.percentile(img, 100 - percentile, axis=(0, 1), keepdims=True)
        img = np.clip(img, lo, hi)

        # Minkowski norm per channel (shades of gray, p=6).
        power = np.power(np.maximum(img, 1e-6), p).mean(axis=(0, 1))
        norm = np.power(power, 1.0 / p)
        gray = float(norm.mean())
        gains = gray / np.maximum(norm, 1e-6)
        gains = np.clip(gains, 0.4, 2.5)  # stability clamp

        out = img * gains.reshape(1, 1, 3)
        return self._same_size(image, np.clip(out, 0, 255).astype(np.uint8))
