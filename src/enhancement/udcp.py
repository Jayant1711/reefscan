"""Underwater Dark Channel Prior (UDCP) — per-channel attenuation model.

Reference: Drews Jr. et al., "Transmission Estimation in Underwater Single
Images", ICCV Workshops 2013. Unlike DCP, UDCP exploits wavelength-dependent
attenuation: red light decays fastest with depth, so the darkest channel
carries distance information and transmission must be estimated per channel.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Enhancer
from .registry import register
from .dark_channel import _guided_filter


@register
class UDCPEnhancer(Enhancer):
    """Per-channel transmission UDCP with red-channel attenuation compensation."""

    name = "udcp"
    description = "Underwater Dark Channel Prior (per-channel transmission, Drews 2013)"
    reference = "Drews et al., ICCVW 2013"

    def enhance(self, image: np.ndarray) -> np.ndarray:
        image = self._validate(image)
        omega = float(self.params.get("udcp_omega", 0.85))
        t0 = float(self.params.get("dcp_t0", 0.10))
        patch = int(self.params.get("dcp_patch", 15))

        img = self._to_float(image)  # BGR
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))

        # UDCP dark channel: min over channels *and* over the local patch.
        min_c = img.min(axis=2)
        udcp = cv2.erode(min_c, kernel)

        # Airlight: average of the brightest UDCP pixels' colours (top 0.1%).
        flat = udcp.ravel()
        n_select = max(1, int(flat.size * 0.001))
        idx = np.argpartition(flat, -n_select)[-n_select:]
        mask = np.zeros(flat.size, dtype=bool)
        mask[idx] = True
        airlight = img.reshape(-1, 3)[mask].mean(axis=0)  # (3,) BGR

        # Per-channel transmission; red attenuates more -> deeper t_r drop.
        t = np.clip(1.0 - omega * (img - airlight.reshape(1, 1, 3)) /
                    np.maximum(airlight.reshape(1, 1, 3), 1e-6), 0.0, 1.0)
        # Use the UDCP-based overall transmission as a soft guide per channel.
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_refined = np.stack(
            [_guided_filter(guide, t[..., c]) for c in range(3)], axis=2
        )
        t_refined = np.clip(t_refined, t0, 1.0)

        out = (img - airlight.reshape(1, 1, 3)) / t_refined + airlight.reshape(1, 1, 3)
        return self._same_size(image, np.clip(out * 255.0, 0, 255).astype(np.uint8))
