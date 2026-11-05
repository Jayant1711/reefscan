"""CLAHE — Contrast Limited Adaptive Histogram Equalization on the L channel of LAB."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register


@register
class CLAHEEnhancer(Enhancer):
    """Contrast Limited AHE (Zuiderveld, 1994) applied to luminance only, so hue is preserved."""

    name = "clahe"
    description = "Contrast Limited Adaptive Histogram Equalization on LAB luminance"
    reference = "Zuiderveld, 1994"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        clip = float(self.params.get("clahe_clip_limit", 2.0))
        tile = int(self.params.get("clahe_tile", 8))

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        lab[..., 0] = clahe.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return self._same_size(image, out)
