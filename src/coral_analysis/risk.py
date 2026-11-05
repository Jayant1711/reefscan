"""Coral bleaching risk scoring (Phase 3 + Phase 6 environmental risk input)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskProfile:
    score: float                 # 0-100
    category: str                # Low / Moderate / High / Critical
    drivers: list[str]
    actions: list[str]


def coral_risk_score(
    mean_severity_index: float,
    bleached_frame_fraction: float = 0.0,
    turbidity_proxy: float = 0.0,
    severity_trend_slope: float = 0.0,
    severity_weight: float = 0.50,
    coverage_weight: float = 0.20,
    environment_weight: float = 0.15,
    trend_weight: float = 0.15,
) -> RiskProfile:
    """Composite bleaching risk in [0, 100].

    Components (all normalised to 0-100):
      severity   : mean BSI across patches
      coverage   : fraction of recent frames showing any bleaching
      environment: turbidity proxy (0-1 scaled)
      trend      : BSI slope per survey step, squashed via tanh
    """
    sev = float(np.clip(mean_severity_index, 0, 100))
    cov = 100.0 * float(np.clip(bleached_frame_fraction, 0, 1))
    env = 100.0 * float(np.clip(turbidity_proxy, 0, 1))
    trend = 100.0 * float(np.tanh(severity_trend_slope))

    score = (
        severity_weight * sev
        + coverage_weight * cov
        + environment_weight * env
        + trend_weight * trend
    )
    score = float(np.clip(score, 0, 100))
    cat = risk_category(score)

    drivers: list[str] = []
    if sev >= 35:
        drivers.append(f"Elevated bleaching severity (BSI {sev:.0f}/100)")
    if cov >= 50:
        drivers.append(f"Bleaching visible in {cov:.0f}% of analysed frames")
    if env >= 50:
        drivers.append("High turbidity / low water clarity")
    if severity_trend_slope > 2:
        drivers.append("Deteriorating severity trend")
    if not drivers:
        drivers.append("No dominant risk drivers detected")

    actions = {
        "Low": ["Keep routine monitoring cadence."],
        "Moderate": ["Shorten monitoring interval; log temperature anomalies."],
        "High": ["Deploy rapid-response survey; remove local stressors."],
        "Critical": ["Immediate management notification; plan restoration."],
    }[cat]
    return RiskProfile(score=score, category=cat, drivers=drivers, actions=actions)


def risk_category(score: float) -> str:
    """Map a 0-100 risk score to a categorical label."""
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"
