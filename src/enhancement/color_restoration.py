"""Color restoration: red-channel compensation + per-channel contrast stretch.

Water absorbs ~a 10x more red than blue per metre, so even white-balanced
images stay red-starved. This strategy re-inflates the red channel relative to
G/B, then clips extreme percentiles per channel for vivid, balanced colour.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register


@register
class ColorRestorationEnhancer(Enhancer):
    name = "color_restoration"
    description = "Red-channel compensation + percentile contrast stretching"
    reference = "Galdran et al., 2015 (red channel prior inspired)"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        red_gain = float(self.params.get("cr_red_gain", 1.15))
        percentile = float(self.params.get("wb_percentile", 2.0))

        img = image.astype(np.float32)

        # --- Red compensation: pull red toward the green/blue mean.
        mean_gb = img[..., 1:3].mean(axis=2)
        deficit = np.clip(mean_gb - img[..., 2], 0, None)  # R below G/B mean
        out = img.copy()
        out[..., 2] = img[..., 2] + red_gain * deficit

        # --- Per-channel percentile stretch (2..98).
        lo = np.percentile(out, percentile, axis=(0, 1), keepdims=True)
        hi = np.percentile(out, 100 - percentile, axis=(0, 1), keepdims=True)
        scale = np.maximum(hi - lo, 1e-6)
        out = (out - lo) / scale * 255.0

        return self._same_size(image, np.clip(out, 0, 255).astype(np.uint8))
