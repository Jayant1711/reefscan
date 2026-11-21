"""The realtime segmenter must receive the RAW frame, not the enhanced one.

Live-caught on 2026-11-20: the U-Net is trained on raw DeepFish frames, but
the pipeline fed it the auto-enhanced image (color_restoration) — the shifted
intensity statistics saturated the sigmoid, producing a 100%-foreground mask,
a whole-frame box, and scene-level (low-confidence) species probabilities.
On the validation split raw input hits 191/310 frames vs 150 enhanced.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("torch", reason="pipeline needs torch")

from src.config import load_config
from src.realtime.pipeline import ReefScanPipeline


class RecordingSegmenter:
    """Stub segmenter that records exactly what it is fed."""

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict_mask(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        self.inputs.append(image_rgb_uint8.copy())
        return np.zeros(image_rgb_uint8.shape[:2], dtype=np.float32)  # no fish


def test_segmenter_receives_raw_frame_not_enhanced(tmp_path) -> None:
    cfg = load_config()
    pipe = ReefScanPipeline(cfg, device=None, enable_cam=False)
    img = (np.random.RandomState(0).randint(0, 255, (96, 128, 3))).astype("uint8")

    recorder = RecordingSegmenter()
    pipe.segmenter = recorder
    # Drastic enhancement: if the pipeline ever feeds the enhanced frame, the
    # recorder will see brightened pixels and the assertion fails.
    pipe.enhancer.enhance = lambda im: cv2.convertScaleAbs(im, alpha=3.0, beta=60.0)

    pipe.analyze_image(img)

    assert recorder.inputs, "segmenter was never called"
    fed = recorder.inputs[0]
    expected = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert fed.shape == expected.shape
    # The recorded frame must match the RAW input (max diff 0), not the
    # alpha=3.0/beta=60 enhanced variant (max diff would be ~600).
    assert np.abs(fed.astype(int) - expected.astype(int)).max() == 0


def test_classifier_and_coral_receive_raw_frames(tmp_path) -> None:
    """Second skew family (2026-11-21): fish crops, coral input, WGBS whiteness
    and GradCAM were all computed on the ENHANCED frame while every model was
    trained on raw images. Assert the raw pixels reach each model."""
    cfg = load_config()
    pipe = ReefScanPipeline(cfg, device=None, enable_cam=False)
    img = np.random.RandomState(1).randint(0, 255, (96, 128, 3)).astype("uint8")

    captured = []
    orig_predict = pipe._predict

    def recording_predict(model, image_bgr, size):
        captured.append((id(model), image_bgr.copy()))
        n = len(getattr(model, "classes_", []) or []) or 2
        return 0, np.full(max(n, 2), 1.0 / max(n, 2))

    pipe._predict = recording_predict
    recorder = RecordingSegmenter()
    # One fish-sized box on the left half -> classifier crop = raw left half.
    mask = np.zeros((96, 128), dtype=np.float32)
    mask[20:70, 10:60] = 1.0
    pipe.segmenter = type("S", (), {"predict_mask": staticmethod(lambda rgb: mask)})()
    pipe.enhancer.enhance = lambda im: cv2.convertScaleAbs(im, alpha=3.0, beta=60.0)

    analysis = pipe.analyze_image(img)

    # _predict receives BGR (it converts to RGB internally) — compare against
    # the raw BGR frame; the enhanced variant (alpha=3, beta=60) would differ.
    coral_calls = [c for mid, c in captured if mid != id(pipe.fish_model)]
    fish_calls = [c for mid, c in captured if mid == id(pipe.fish_model)]
    assert coral_calls and fish_calls
    assert np.abs(coral_calls[0].astype(int) - img.astype(int)).max() == 0
    # Fish classifier crop = the raw left-half region.
    crop = fish_calls[0]
    assert crop.shape[1] == 50 and crop.shape[0] == 50
    assert np.abs(crop.astype(int) - img[20:70, 10:60].astype(int)).max() == 0
    assert analysis.detections


def test_video_mode_skips_scene_fallback(tmp_path) -> None:
    """Video counting must not invent a phantom whole-frame fish on every
    fishless frame (would inflate unique-ID counts)."""
    cfg = load_config()
    pipe = ReefScanPipeline(cfg, device=None, enable_cam=False)
    img = np.random.RandomState(2).randint(0, 255, (64, 64, 3)).astype("uint8")
    pipe.segmenter = type("S", (), {"predict_mask": staticmethod(
        lambda rgb: np.zeros((64, 64), dtype=np.float32))})()

    a_off = pipe.analyze_image(img, allow_scene_fallback=False)
    assert a_off.detections == [] and not a_off.scene_fallback

    a_on = pipe.analyze_image(img)  # image mode keeps the usable fallback
    assert len(a_on.detections) == 1 and a_on.scene_fallback
