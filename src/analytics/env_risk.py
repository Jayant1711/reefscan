"""Environmental risk assessment from image physics proxies — Phase 6 innovation #5.

Two optical proxies are measured per frame on the *raw* (unenhanced) capture:

- Turbidity  : mean dark-channel intensity. Suspended particulates lift the
               dark channel (haze physics, He et al. 2010), so a high mean
               indicates murkier water.
- Visibility : RMS contrast of the luminance channel; low contrast implies
               strong forward scattering / poor visibility.

Both are squashed to [0, 1] and combined with the bleaching trend slope into
the environmental risk input used by the Reef Health Score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TurbidityMetrics:
    turbidity_proxy: float          # 0 (clear) .. 1 (murky)
    visibility_proxy: float         # 0 (poor) .. 1 (excellent)
    mean_dark_channel: float        # raw diagnostic
    rms_contrast: float             # raw diagnostic


@dataclass
class EnvRiskResult:
    score: float                    # 0-100 (higher = riskier)
    category: str                   # Low / Moderate / High / Critical
    turbidity: TurbidityMetrics
    drivers: list[str] = field(default_factory=list)


def measure_turbidity(image_bgr: np.ndarray, dark_patch: int = 15) -> TurbidityMetrics:
    """Compute turbidity + visibility proxies for one frame."""
    img = image_bgr.astype(np.float32) / 255.0
    min_c = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dark_patch, dark_patch))
    dark = cv2.erode(min_c, kernel)
    mean_dark = float(dark.mean())

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rms_contrast = float(gray.std())

    # Squashing functions (empirically reasonable for 8-bit underwater video).
    turbidity = float(np.clip(mean_dark / 0.35, 0, 1))
    visibility = float(np.clip(rms_contrast / 0.22, 0, 1))
    return TurbidityMetrics(turbidity, visibility, mean_dark, rms_contrast)


def environmental_risk(
    turbidity: TurbidityMetrics,
    severity_trend_slope: float = 0.0,
    turbidity_weight: float = 0.45,
    visibility_weight: float = 0.25,
    trend_weight: float = 0.30,
) -> EnvRiskResult:
    """Composite environmental risk in [0, 100]."""
    turb = 100.0 * turbidity.turbidity_proxy
    vis_penalty = 100.0 * (1.0 - turbidity.visibility_proxy)
    trend = 100.0 * float(np.tanh(severity_trend_slope))

    score = turbidity_weight * turb + visibility_weight * vis_penalty + trend_weight * trend
    score = float(np.clip(score, 0, 100))

    if score < 25:
        category = "Low"
    elif score < 50:
        category = "Moderate"
    elif score < 75:
        category = "High"
    else:
        category = "Critical"

    drivers = []
    if turbidity.turbidity_proxy > 0.5:
        drivers.append("Elevated turbidity (suspended sediment / algal bloom)")
    if turbidity.visibility_proxy < 0.4:
        drivers.append("Poor water visibility (forward scattering)")
    if severity_trend_slope > 2:
        drivers.append("Bleaching severity increasing across the survey")
    if not drivers:
        drivers.append("Water conditions within normal range")
    return EnvRiskResult(score=score, category=category, turbidity=turbidity, drivers=drivers)
