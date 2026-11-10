"""Regression tests pinning the audit fixes (torch-free, run locally)."""

from __future__ import annotations

import json

import pytest

from src.realtime.tracker import CentroidTracker
from src.segmentation.postprocess import BBox
from src.utils.file_utils import save_json


def _box(x1, y1, x2, y2, score=0.9):
    return BBox(x1, y1, x2, y2, score, (x2 - x1) * (y2 - y1))


# ------------------------------------------------------------------ tracker
def test_tracker_all_detections_far_apart_no_crash():
    """Pins the nanargmin fix: every detection beyond min_distance must
    register new tracks instead of raising 'All-NaN slice encountered'."""
    tracker = CentroidTracker(max_disappeared=15, min_distance=40.0)
    boxes = [_box(10, 10, 50, 50), _box(500, 500, 540, 540), _box(1200, 100, 1240, 140)]
    live = tracker.update(boxes, frame_idx=0)
    assert tracker.unique_count == 3
    assert len(live) == 3


def test_tracker_empty_frames_age_and_retire():
    """Pins the aging fix: tracks must retire during detection-free stretches."""
    tracker = CentroidTracker(max_disappeared=3, min_distance=40.0)
    tracker.update([_box(0, 0, 40, 40)], frame_idx=0)
    assert tracker.unique_count == 1
    for frame in range(1, 10):
        live = tracker.update([], frame_idx=frame)
    assert live == []                      # retired, not lingering forever
    assert tracker.retired_counts.get("unknown") == 1
    assert tracker.species_counts().get("unknown") == 1  # still counted at end-of-video
    assert tracker.unique_count == 1                    # unique individuals unchanged


def test_tracker_refound_after_gap_gets_new_id():
    """A fish that left and returns far away is a NEW individual (stale track
    must have been retired, not absorbed)."""
    tracker = CentroidTracker(max_disappeared=2, min_distance=40.0)
    tracker.update([_box(0, 0, 40, 40)], frame_idx=0)
    for frame in range(1, 6):
        tracker.update([], frame_idx=frame)
    tracker.update([_box(600, 600, 640, 640)], frame_idx=6)
    assert tracker.unique_count == 2


# ----------------------------------------------------------------- BBox.iou
def test_bbox_iou_uses_rectangle_union_not_component_area():
    """Pins the IoU fix: identical rectangles overlap perfectly regardless of
    the connected-component pixel count stored in ``area``."""
    gt = BBox(10, 10, 110, 110, 1.0, 10_000)            # solid blob
    thin_pred = BBox(10, 10, 110, 110, 0.9, 300)        # same rect, thin blob
    assert thin_pred.iou(gt) == pytest.approx(1.0)


def test_bbox_iou_half_overlap():
    a = _box(0, 0, 100, 100)                # area 10_000
    b = _box(50, 0, 150, 100)               # overlap 50x100 = 5000
    assert a.iou(b) == pytest.approx(5000 / 15000)


# ---------------------------------------------------------------- save_json
def test_save_json_non_finite_becomes_null(tmp_path):
    """Pins the strict-JSON fix: NaN/inf serialize as null, not bare NaN."""
    out = tmp_path / "metrics.json"
    save_json({"macro_ap": float("nan"), "iou": float("inf"), "ok": 0.5,
              "nested": {"pixel_ap": [float("nan"), 1.0]}}, out)
    text = out.read_text(encoding="utf-8")

    def _reject(token):  # strict parser: any NaN/Infinity literal raises
        raise ValueError(f"non-JSON constant: {token}")

    data = json.loads(text, parse_constant=_reject)     # must not raise
    assert data["macro_ap"] is None and data["iou"] is None
    assert data["nested"]["pixel_ap"][0] is None
    assert data["ok"] == 0.5 and data["nested"]["pixel_ap"][1] == 1.0


# --------------------------------------------------------- detect_platform
def test_detect_platform_lightning_filesystem_fallback(monkeypatch):
    """Pins the Lightning fallback: studios without env vars still detect."""
    import src.utils.provenance as prov

    from os import path as _ospath

    real_exists = _ospath.exists

    def fake_exists(path):
        return str(path) == "/teamspace" or real_exists(path)

    monkeypatch.setattr("os.path.exists", fake_exists)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("LIGHTNING_CLOUD_PROJECT_ID", raising=False)
    monkeypatch.delenv("LIGHTNING_API_TOKEN", raising=False)
    assert prov.detect_platform() == "lightning"


# ------------------------------------------------- last-stage regression pin
def test_last_stage_prefixes_exist_in_torchvision_names():
    """The prefixes must match real parameter names — stages.3 matched zero
    params on ConvNeXt (the original audit bug). Torch-free: static check."""

    from src.classification.models import _LAST_STAGE

    expected = {
        "resnet34": "layer4",
        "efficientnet_b0": "features.7",
        "mobilenet_v3": "features.12",
        "convnext_tiny": "features.7",
    }
    assert _LAST_STAGE == expected
    # (torchvision name verification runs with the full suite on GPU platforms;
    #  locally the DL stack is policy-blocked, so only the static map is checked.)
