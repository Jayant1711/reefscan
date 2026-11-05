"""Shared fixtures: synthetic images and a tiny dataset builder."""

from __future__ import annotations

import cv2
import numpy as np
import pytest


@pytest.fixture
def gradient_bgr() -> np.ndarray:
    """Deterministic 96x96 BGR gradient with structure (non-flat)."""
    x = np.linspace(0, 255, 96, dtype=np.float32)
    xx, yy = np.meshgrid(x, x)
    b = (0.3 * xx + 0.2 * yy).astype(np.uint8)
    g = (0.5 * xx).astype(np.uint8)
    r = (0.4 * yy).astype(np.uint8)
    return cv2.merge([b, g, r])


@pytest.fixture
def flat_bgr() -> np.ndarray:
    return np.full((96, 96, 3), 128, dtype=np.uint8)


@pytest.fixture
def pale_white_bgr() -> np.ndarray:
    """Bleached-looking: bright, low saturation."""
    hsv = np.zeros((96, 96, 3), dtype=np.uint8)
    hsv[..., 0] = 30
    hsv[..., 1] = 20          # very low saturation
    hsv[..., 2] = 220         # very bright
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.fixture
def saturated_bgr() -> np.ndarray:
    """Healthy-looking: strong saturation, mid value."""
    hsv = np.zeros((96, 96, 3), dtype=np.uint8)
    hsv[..., 0] = 15
    hsv[..., 1] = 200
    hsv[..., 2] = 150
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
