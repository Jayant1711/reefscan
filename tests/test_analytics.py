"""Analytics tests: biodiversity (hand-computed), reef health, env risk, tracker."""

from __future__ import annotations

import math

import numpy as np

from src.analytics.biodiversity import (
    gini_simpson_index,
    margalef_richness,
    marine_biodiversity_score,
    shannon_index,
)
from src.analytics.env_risk import environmental_risk, measure_turbidity
from src.analytics.reef_health import reef_health_score
from src.realtime.tracker import CentroidTracker
from src.segmentation.postprocess import BBox, masks_to_boxes


# ---------------------------------------------------------------- biodiversity
def test_shannon_two_uniform_species():
    # H' = -(0.5 ln 0.5 + 0.5 ln 0.5) = ln 2
    counts = {"A": 10, "B": 10}
    assert abs(shannon_index(counts) - math.log(2)) < 1e-9


def test_shannon_zero_diversity():
    assert shannon_index({"A": 50}) == 0.0


def test_gini_simpson_two_uniform():
    # 1 - (0.25 + 0.25) = 0.5
    assert abs(gini_simpson_index({"A": 1, "B": 1}) - 0.5) < 1e-9


def test_margalef_formula():
    # d = (S-1)/ln N with S=3, N=9 -> 2/ln9
    counts = {"A": 4, "B": 3, "C": 2}
    assert abs(margalef_richness(counts) - 2 / math.log(9)) < 1e-9


def test_mbs_bounds_and_ordering():
    low = marine_biodiversity_score({"A": 100})
    high = marine_biodiversity_score({f"S{i}": 10 for i in range(10)})
    assert 0 <= low["mbs"] <= 100 and 0 <= high["mbs"] <= 100
    assert high["mbs"] > low["mbs"]
    assert marine_biodiversity_score({})["mbs"] == 0.0


# ----------------------------------------------------------------- reef health
def test_reef_health_score_math():
    # 100 - (0.6*50 + 0.25*0 + 0.15*0) = 70 -> Good
    res = reef_health_score(mean_severity_index=50)
    assert abs(res.score - 70.0) < 1e-9
    assert res.category == "Good"

    pristine = reef_health_score(0, 0.0, 0.0)
    assert pristine.score == 100.0 and pristine.category == "Excellent"

    worst = reef_health_score(100, 1.0, 100.0)
    assert worst.score == 0.0 and worst.category == "Critical"


# ------------------------------------------------------------------ env risk
def test_measure_turbidity_ordering():
    clear = np.full((64, 64, 3), 255, dtype=np.uint8)
    clear[..., 2] = 10          # blue-dominant bright
    murky = np.full((64, 64, 3), 70, dtype=np.uint8)
    t_clear = measure_turbidity(clear)
    t_murky = measure_turbidity(murky)
    assert t_murky.mean_dark_channel > t_clear.mean_dark_channel


def test_environmental_risk_categories():
    from src.analytics.env_risk import TurbidityMetrics

    good = TurbidityMetrics(0.05, 0.9, 0.02, 0.2)
    bad = TurbidityMetrics(0.9, 0.05, 0.3, 0.01)
    assert environmental_risk(good).category in {"Low", "Moderate"}
    assert environmental_risk(bad, severity_trend_slope=5.0).score > environmental_risk(good).score


# -------------------------------------------------------------------- tracker
def test_centroid_tracker_counts_unique():
    tracker = CentroidTracker(max_disappeared=3, min_distance=30)
    # Same fish drifting slowly -> one track.
    for shift in range(6):
        box = BBox(10 + shift, 10, 40 + shift, 30, 0.9, 400)
        tracker.update([box], frame_idx=shift)
    assert tracker.unique_count == 1
    # A second, far-apart fish appears -> second track.
    tracker.update([BBox(10, 10, 40, 30, 0.9, 400), BBox(150, 150, 190, 180, 0.9, 900)], frame_idx=10)
    assert tracker.unique_count == 2
    assert tracker.species_counts()


def test_postprocess_bboxes():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[10:30, 10:30] = 0.9
    mask[40:50, 40:55] = 0.5
    boxes = masks_to_boxes(mask, threshold=0.4, min_area=50)
    assert len(boxes) == 2
    big = boxes[0]
    assert (big.x1, big.y1, big.x2, big.y2) == (10, 10, 30, 30)
    assert abs(big.score - 0.9) < 1e-6
    # Rescaling back to a larger original size.
    scaled = masks_to_boxes(mask, threshold=0.4, min_area=50, original_size=(128, 128))
    assert scaled[0].x2 == 60 and scaled[0].y2 == 60
