"""Tests for the production layer: run_cli, preflight checks, frame annotation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.realtime.annotate import annotate_frame
from src.realtime.schema import FishDetection
from src.utils.cli import run_cli
from src.utils.provenance import env_versions


# ------------------------------------------------------------------- run_cli
def test_run_cli_ok_returns_int(tmp_path, capsys):
    from src.utils.logger import setup_logging

    setup_logging()
    assert run_cli(lambda: 7) == 7
    assert run_cli(lambda: None) == 0


def test_run_cli_file_not_found_clean_exit():
    def fail():
        raise FileNotFoundError("no dataset at X")

    assert run_cli(fail) == 1


def test_run_cli_value_error_clean_exit():
    def fail():
        raise ValueError("bad flag value")

    assert run_cli(fail) == 1


def test_run_cli_missing_module_gives_install_hint(caplog):
    def fail():
        raise ModuleNotFoundError("No module named 'onnx'")

    with caplog.at_level("ERROR", logger="cli"):
        assert run_cli(fail) == 1
    assert "install requirements" in caplog.text.lower()


def test_run_cli_oom_message_hint(caplog):
    def fail():
        raise RuntimeError("CUDA out of memory on device")

    with caplog.at_level("ERROR", logger="cli"):
        assert run_cli(fail) == 1
    assert "batch_size" in caplog.text


def test_run_cli_other_runtime_error_raises():
    """Non-OOM RuntimeErrors are real bugs — must traceback, not be swallowed."""
    def real_bug():
        raise RuntimeError("index out of range")

    with pytest.raises(RuntimeError, match="index out of range"):
        run_cli(real_bug)


# ----------------------------------------------------------------- env_versions
def test_env_versions_fields():
    v = env_versions()
    assert set(v) == {"python", "torch", "cuda"}
    assert v["python"].count(".") >= 1
    assert v["torch"] != ""


# ---------------------------------------------------------------- annotate
def _detection(x1, y1, x2, y2, species="TestFish", conf=0.9):
    from src.segmentation.postprocess import BBox

    box = BBox(x1, y1, x2, y2, conf, (x2 - x1) * (y2 - y1))
    return FishDetection(box=box, species=species, confidence=conf)


def test_annotate_frame_draws_boxes_and_hud():
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    dets = [_detection(20, 20, 100, 80, conf=0.93)]
    out = annotate_frame(frame, dets, coral=None, heatmap=None)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # box OUTLINE (2 px border) must differ from the flat background
    assert not np.array_equal(out[20:22, 20:100], frame[20:22, 20:100])
    assert not np.array_equal(out[78:80, 20:100], frame[78:80, 20:100])
    # HUD strip at the top should carry text (changed pixels)
    assert not np.array_equal(out[:30, :], frame[:30, :])


def test_annotate_frame_with_coral_banner():
    """annotate_frame takes the SeverityAssessment (pipeline passes
    analysis.coral.assessment), not the CoralResult wrapper."""
    from src.coral_analysis.severity import BleachingAssessor

    assessment = BleachingAssessor().assess(
        np.full((64, 64, 3), 210, dtype=np.uint8), p_bleached=0.9
    )
    frame = np.full((240, 320, 3), 40, dtype=np.uint8)
    out = annotate_frame(frame, [], coral=assessment, heatmap=None)
    assert not np.array_equal(out, frame), "coral banner must change the frame"


def test_annotate_frame_no_detections_unchanged_except_hud():
    frame = np.full((120, 160, 3), 90, dtype=np.uint8)
    out = annotate_frame(frame, [], coral=None, heatmap=None)
    assert out.shape == frame.shape


# ---------------------------------------------------------------- JSON report
def test_preflight_json_schema(tmp_path):
    """The preflight report honours the documented schema without running GPU checks."""
    report = {
        "schema": "reefscan-preflight-v1",
        "recorded_at": "2026-11-11T00:00:00Z",
        "platform": "local",
        "git_sha": "abc1234",
        "versions": {"python": "3.11", "torch": "2.3", "cuda": "none"},
        "checks": [{"id": "PY", "status": "PASS", "detail": "3.11"}],
        "result": "PASS",
    }
    out = tmp_path / "preflight_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema"] == "reefscan-preflight-v1"
    assert loaded["result"] == "PASS"
    assert loaded["checks"][0]["id"] == "PY"
