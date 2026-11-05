"""Composite pipeline: white balance -> CLAHE -> adaptive gamma.

Underwater scenes benefit from colour constancy *before* local contrast
enhancement *before* exposure normalisation. This cheap 3-stage fusion is the
strongest classical baseline in our benchmark on most inputs.
"""

from __future__ import annotations

import numpy as np

from .base import Enhancer
from .registry import register
from .white_balance import WhiteBalanceEnhancer
from .clahe import CLAHEEnhancer
from .gamma import GammaCorrectionEnhancer


@register
class FusionWCGEnhancer(Enhancer):
    name = "fusion_wcg"
    description = "Fusion: white balance -> CLAHE -> adaptive gamma"
    reference = "Composite of Bianco 2017 + Zuiderveld 1994 + Huang 2012"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self._wb = WhiteBalanceEnhancer(params)
        self._clahe = CLAHEEnhancer(params)
        self._gamma = GammaCorrectionEnhancer(params)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        stage1 = self._wb.enhance(image)
        stage2 = self._clahe.enhance(stage1)
        out = self._gamma.enhance(stage2)
        return self._same_size(image, out)
