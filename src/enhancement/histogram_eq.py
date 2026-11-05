"""Global histogram equalization on luminance."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register


@register
class HistogramEqualizationEnhancer(Enhancer):
    """Classical global HE — a baseline that often over-amplifies noise underwater."""

    name = "histogram_eq"
    description = "Global histogram equalization on luminance (baseline)"
    reference = "Gonzalez & Woods, Digital Image Processing"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.equalizeHist(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return self._same_size(image, out)
