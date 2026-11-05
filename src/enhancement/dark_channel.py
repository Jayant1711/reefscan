"""Dark Channel Prior haze removal (He et al., 2010) with guided-filter refinement."""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register


def _dark_channel(img_float: np.ndarray, patch: int) -> np.ndarray:
    """Per-pixel minimum across channels, min-filtered over a patch."""
    min_c = img_float.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_c, kernel)


def _guided_filter(im: np.ndarray, p: np.ndarray, radius: int = 60, eps: float = 1e-3) -> np.ndarray:
    """Edge-preserving guided filter (He et al., 2013), grayscale guided."""
    mean_i = cv2.boxFilter(im, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1))
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1))
    mean_ip = cv2.boxFilter(im * p, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1))
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(im * im, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1))
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1))
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=(2 * radius + 1, 2 * radius + 1))
    return mean_a * im + mean_b


@register
class DarkChannelPriorEnhancer(Enhancer):
    """Classic DCP dehazing adapted to underwater scenes.

    1. dark channel over a 15x15 patch,
    2. airlight A from the top 0.1% brightest dark-channel pixels,
    3. transmission t = 1 - omega * darkchannel / A, refined by guided filter,
    4. scene recovery J = (I - A) / max(t, t0) + A.
    """

    name = "dark_channel"
    description = "Dark Channel Prior dehazing + guided filter transmission refinement"
    reference = "He, Sun & Tang, CVPR 2010 / TPAMI 2011"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        patch = int(self.params.get("dcp_patch", 15))
        omega = float(self.params.get("dcp_omega", 0.95))
        t0 = float(self.params.get("dcp_t0", 0.10))

        img = self._to_float(image)
        dark = _dark_channel(img, patch)

        # --- Airlight estimation: brightest pixels among the haziest regions.
        flat_dark = dark.ravel()
        n_select = max(1, int(flat_dark.size * 0.001))
        hazy_idx = np.argpartition(flat_dark, -n_select)[-n_select:]
        mask = np.zeros_like(dark, dtype=bool).ravel()
        mask[hazy_idx] = True
        mask = mask.reshape(dark.shape)
        airlight = img[mask].max(axis=0)  # per-channel airlight (BGR)

        # --- Transmission + guided refinement on the grayscale guide.
        transmission = 1.0 - omega * (dark / max(airlight.max(), 1e-6))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        transmission = _guided_filter(gray, transmission)
        transmission = np.clip(transmission, t0, 1.0)

        # --- Scene radiance recovery.
        out = (img - airlight.reshape(1, 1, 3)) / transmission[..., None] + airlight.reshape(1, 1, 3)
        return self._same_size(image, np.clip(out * 255.0, 0, 255).astype(np.uint8))
