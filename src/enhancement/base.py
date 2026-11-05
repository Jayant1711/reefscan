"""Abstract base class for all enhancement strategies.

Contract: BGR uint8 in -> BGR uint8 out, same spatial size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from ..utils.logger import get_logger


class Enhancer(ABC):
    """Base class for underwater image enhancement strategies."""

    #: unique registry name
    name: str = "base"
    #: human-readable description used in reports / dashboard
    description: str = ""
    #: literature reference
    reference: str = ""

    def __init__(self, params: dict | None = None) -> None:
        self.params: dict = dict(params or {})
        self.logger = get_logger(f"enhance.{self.name}")

    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a BGR uint8 image and return a BGR uint8 image of the same size."""

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.enhance(image)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _validate(image: np.ndarray) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Enhancers expect a 3-channel BGR image.")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _to_float(image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    @staticmethod
    def _clip_uint8(image: np.ndarray) -> np.ndarray:
        return np.clip(image * 255.0 if image.max() <= 1.0 + 1e-6 and image.dtype != np.uint8 else image, 0, 255).astype(np.uint8)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Enhancer {self.name}>"

    def _same_size(self, original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        if enhanced.shape != original.shape:
            enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        return enhanced
