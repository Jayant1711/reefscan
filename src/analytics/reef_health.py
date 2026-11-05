"""Reef Health Score (RHS) — Phase 6 innovation #2.

    RHS = 100 - ( w_sev  * mean_BSI
                 + w_cov  * 100 * bleached_frame_fraction
                 + w_risk * environmental_risk )

Default weights: severity 0.60, coverage 0.25, risk 0.15.
Category bands: >=80 Excellent, >=60 Good, >=40 Fair, >=20 Poor, else Critical.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_WEIGHTS = {"severity": 0.60, "bleached_frac": 0.25, "risk": 0.15}

BANDS = [
    (80, "Excellent"),
    (60, "Good"),
    (40, "Fair"),
    (20, "Poor"),
    (0, "Critical"),
]


@dataclass
class ReefHealthResult:
    score: float
    category: str
    components: dict[str, float] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.category


def reef_health_score(
    mean_severity_index: float,
    bleached_frame_fraction: float = 0.0,
    environmental_risk_score: float = 0.0,
    weights: dict[str, float] | None = None,
) -> ReefHealthResult:
    """Compute the composite Reef Health Score (0 = dead reef, 100 = pristine)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    sev = float(np.clip(mean_severity_index, 0, 100))
    cov = 100.0 * float(np.clip(bleached_frame_fraction, 0, 1))
    risk = float(np.clip(environmental_risk_score, 0, 100))

    penalty = w["severity"] * sev + w["bleached_frac"] * cov + w["risk"] * risk
    score = float(np.clip(100.0 - penalty, 0, 100))
    category = next(label for floor, label in BANDS if score >= floor)
    return ReefHealthResult(
        score=score,
        category=category,
        components={"mean_severity": sev, "bleached_frame_pct": cov, "env_risk": risk},
    )
