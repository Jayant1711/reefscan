"""Adaptive gamma correction driven by the image mean luminance."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register


@register
class GammaCorrectionEnhancer(Enhancer):
    """Gamma = log(target_mean) / log(observed_mean), clamped to [0.3, 2.5].

    Brightens dark green/blue underwater scenes and gently tames blown-out
    shallow shots, without touching colour ratios.
    """

    name = "gamma"
    description = "Adaptive gamma correction from mean luminance"
    reference = "Huang et al., 2012 (adaptive gamma)"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        target = float(self.params.get("gamma_mean_target", 0.5))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean = float(gray.mean())
        mean = min(max(mean, 1e-3), 1.0 - 1e-3)
        gamma = float(np.log(target) / np.log(mean))
        gamma = float(np.clip(gamma, 0.3, 2.5))

        table = (np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0).astype(np.uint8)
        out = cv2.LUT(image, table)
        return self._same_size(image, out)
