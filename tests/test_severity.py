"""WGBS severity, risk and trend tests (hand-computed values)."""

from __future__ import annotations

import numpy as np
import pytest

from src.coral_analysis.risk import coral_risk_score, risk_category
from src.coral_analysis.severity import (
    SEVERITY_WEIGHTS,
    BleachingAssessor,
    index_to_class_strict,
    severity_index,
    whiteness_ratio,
)
from src.coral_analysis.trend import BleachingTrendAnalyzer


# ------------------------------------------------------------------- WGBS core
def test_whiteness_ratio_extremes(pale_white_bgr, saturated_bgr):
    assert whiteness_ratio(pale_white_bgr) > 0.9       # pale & bright -> almost all
    assert whiteness_ratio(saturated_bgr) < 0.1        # saturated -> almost none


def test_severity_index_formula():
    # BSI = 100 * (0.6 * p + 0.4 * w)
    assert severity_index(0.0, 0.0) == 0.0
    assert severity_index(1.0, 1.0) == 100.0
    assert abs(severity_index(0.5, 0.5) - 50.0) < 1e-9
    assert abs(severity_index(1.0, 0.0) - 60.0) < 1e-9


def test_index_to_class_boundaries():
    cases = [(0, "healthy"), (14.9, "healthy"), (15, "mild"), (34.9, "mild"),
             (35, "moderate"), (59.9, "moderate"), (60, "severe"), (84.9, "severe"),
             (85, "dead"), (100, "dead")]
    for index, expected in cases:
        assert index_to_class_strict(index) == expected, index


def test_assessor_orders_conditions(pale_white_bgr, saturated_bgr):
    assessor = BleachingAssessor()
    dead_like = assessor.assess(pale_white_bgr, p_bleached=0.99)
    healthy_like = assessor.assess(saturated_bgr, p_bleached=0.01)
    assert dead_like.index > healthy_like.index
    assert dead_like.class_name in {"severe", "dead"}
    assert healthy_like.class_name in {"healthy", "mild"}
    assert dead_like.class_weights_index == SEVERITY_WEIGHTS[dead_like.class_name]
    assert dead_like.recommendations  # non-empty


def test_assessor_distribution():
    assessor = BleachingAssessor()
    a = assessor.assess(np.full((32, 32, 3), 240, np.uint8), 0.95)
    b = assessor.assess(np.zeros((32, 32, 3), np.uint8), 0.02)
    summary = assessor.assess_distribution([a, b])
    assert 0 <= summary["mean_index"] <= 100
    assert abs(sum(summary["distribution"].values()) - 1.0) < 1e-9
    assert summary["n_patches"] == 2


# ----------------------------------------------------------------------- risk
def test_risk_category_bands():
    assert risk_category(0) == "Low"
    assert risk_category(30) == "Moderate"
    assert risk_category(60) == "High"
    assert risk_category(90) == "Critical"


def test_coral_risk_score_monotonic():
    low = coral_risk_score(mean_severity_index=5).score
    high = coral_risk_score(mean_severity_index=95, bleached_frame_fraction=0.9).score
    assert high > low
    assert 0 <= low <= 100 and 0 <= high <= 100


# ---------------------------------------------------------------------- trend
def test_trend_labels():
    rising = BleachingTrendAnalyzer()
    for v in [10, 25, 40, 60, 80]:
        rising.update(v)
    # last >= 60 and slope >= 0 -> "Critical" per the documented labelling rules.
    assert rising.summary().trend_label == "Critical"

    falling = BleachingTrendAnalyzer()
    for v in [80, 60, 40, 25, 10]:
        falling.update(v)
    assert falling.summary().trend_label == "Improving"

    flat = BleachingTrendAnalyzer()
    for v in [50, 50, 50, 50]:
        flat.update(v)
    assert flat.summary().trend_label == "Stable"
