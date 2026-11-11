"""Whiteness-Guided Bleaching Severity (WGBS) — the core ReefScan innovation.

Problem: public coral datasets (incl. NOAA PIFSC ESD) label bleaching *binary*
(CORAL vs CORAL_BL), yet reef managers need graded severity.

Method: fuse the learned bleached-probability with a physics-motivated,
parameter-free whiteness measurement of the patch:

    W = |{p : S(p) < S_max  and  V(p) > V_min}| / |patch|       (HSV space)

    BSI = lambda * 100 * P(bleached | image)  +  (1 - lambda) * 100 * W

Bleached coral loses symbiotic zooxanthellae and its tissue turns pale-white:
high brightness, low saturation. Healthy coral tissue is strongly pigmented;
dead coral is typically overgrown (darker). WGBS therefore reads the same
physical evidence a human annotator uses, but at pixel level, and blends it
with the CNN's context understanding.

BSI (0-100) is mapped to 5 ordinal classes:

    Healthy  [0, 15) | Mild [15, 35) | Moderate [35, 60) | Severe [60, 85) | Dead [85, 100]

The class expectations are also exposed as index weights (0/25/50/75/100) so
the expected severity of a *distribution* of patches can be computed directly.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field

SEVERITY_CLASSES = ["healthy", "mild", "moderate", "severe", "dead"]
SEVERITY_WEIGHTS = {"healthy": 0, "mild": 25, "moderate": 50, "severe": 75, "dead": 100}

# Single source for the default WGBS class cut-offs (mirrors configs/default.yaml
# coral.severity.thresholds; a pinning test keeps the two in sync).
DEFAULT_THRESHOLDS = (15, 35, 60, 85)


def whiteness_ratio(image_bgr: np.ndarray, s_max: int = 70, v_min: int = 140) -> float:
    """Fraction of pale-bright pixels (bleached tissue proxy) in [0, 1].

    Parameters
    ----------
    image_bgr:
        BGR uint8 image.
    s_max:
        HSV saturation threshold below which a pixel is 'pale'.
    v_min:
        HSV value threshold above which a pixel is 'bright/exposed'.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    pale = (s < s_max) & (v > v_min)
    return float(pale.mean())


def severity_index(p_bleached: float, whiteness: float, fusion_lambda: float = 0.60) -> float:
    """Continuous bleaching severity index in [0, 100]."""
    p = float(np.clip(p_bleached, 0.0, 1.0))
    w = float(np.clip(whiteness, 0.0, 1.0))
    lam = float(np.clip(fusion_lambda, 0.0, 1.0))
    return 100.0 * (lam * p + (1.0 - lam) * w)


def index_to_class(index: float, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> str:
    """Map a continuous severity index to one of the 5 ordinal classes."""
    if index < thresholds[0]:
        return "healthy"
    if index < thresholds[1]:
        return "mild"
    if index < thresholds[2]:
        return "moderate"
    if index < thresholds[3]:
        return "severe"
    return "dead"


def index_to_class_strict(index: float, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS) -> str:
    """Explicit (readable) threshold walk for reports and tests."""
    if index < thresholds[0]:
        return "healthy"
    if index < thresholds[1]:
        return "mild"
    if index < thresholds[2]:
        return "moderate"
    if index < thresholds[3]:
        return "severe"
    return "dead"


@dataclass
class SeverityAssessment:
    """Result of assessing one coral patch."""

    index: float                       # BSI 0-100
    class_name: str                    # healthy/mild/moderate/severe/dead
    p_bleached: float                  # CNN probability
    whiteness: float                   # physics proxy
    class_weights_index: int           # expected severity weight of the class
    recommendations: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.class_name.capitalize()


class BleachingAssessor:
    """Fuses a trained coral CNN's bleached-probability with patch whiteness."""

    def __init__(self, fusion_lambda: float = 0.60, thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
                 s_max: int = 70, v_min: int = 140) -> None:
        self.fusion_lambda = fusion_lambda
        self.thresholds = thresholds
        self.s_max = s_max
        self.v_min = v_min

    def assess(self, image_bgr: np.ndarray, p_bleached: float) -> SeverityAssessment:
        """Assess one patch given the model's P(bleached)."""
        w = whiteness_ratio(image_bgr, self.s_max, self.v_min)
        bsi = severity_index(p_bleached, w, self.fusion_lambda)
        cls = index_to_class_strict(bsi, self.thresholds)
        return SeverityAssessment(
            index=bsi,
            class_name=cls,
            p_bleached=float(p_bleached),
            whiteness=w,
            class_weights_index=SEVERITY_WEIGHTS[cls],
            recommendations=_recommendations(cls, bsi),
        )

    def assess_distribution(self, assessments: list[SeverityAssessment]) -> dict:
        """Aggregate several patch assessments into a survey-level summary."""
        if not assessments:
            return {"mean_index": 0.0, "dominant_class": "healthy", "distribution": {}}
        indices = np.array([a.index for a in assessments])
        classes = [a.class_name for a in assessments]
        dist = {c: classes.count(c) / len(classes) for c in SEVERITY_CLASSES}
        expected = sum(SEVERITY_WEIGHTS[c] * frac for c, frac in dist.items())
        dominant = max(dist, key=dist.get)
        return {
            "mean_index": float(indices.mean()),
            "expected_index": float(expected),
            "dominant_class": dominant,
            "distribution": dist,
            "n_patches": len(assessments),
        }


def _recommendations(cls: str, index: float) -> list[str]:
    """Rule-based management recommendations per severity class."""
    base = {
        "healthy": [
            "Continue routine monitoring (monthly photographic surveys).",
            "Maintain protection from anchoring and physical contact.",
        ],
        "mild": [
            "Increase monitoring frequency (bi-weekly).",
            "Verify water temperature anomalies (NOAA Coral Reef Watch DHW).",
        ],
        "moderate": [
            "Flag site for targeted follow-up within 2 weeks.",
            "Reduce local stressors: anchor damage, sediment runoff, fishing pressure.",
        ],
        "severe": [
            "Trigger rapid-response assessment protocol.",
            "Consider shade cloth / assisted recovery interventions if sustained.",
        ],
        "dead": [
            "Record for reef restoration planning (larval seeding, coral gardening).",
            "Investigate compounding factors (disease, predation, pollution).",
        ],
    }
    recs = list(base.get(cls, []))
    if index >= 60:
        recs.append("Notify regional reef-management authority.")
    return recs
