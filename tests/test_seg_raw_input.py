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
