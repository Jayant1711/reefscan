"""Segmentation evaluation tests: greedy matching, metrics on a synthetic checkpoint."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.segmentation.evaluate_seg import _greedy_match
from src.segmentation.engine_seg import dice_iou_scores
from src.segmentation.postprocess import BBox, masks_to_boxes


# ------------------------------------------------------------ greedy matching
def _box(x1, y1, x2, y2, score=0.9):
    return BBox(x1, y1, x2, y2, score, (x2 - x1) * (y2 - y1))


def test_greedy_match_perfect_overlap_is_tp():
    gt = [_box(10, 10, 40, 40)]
    pred = [_box(10, 10, 40, 40, score=0.99)]
    matches = _greedy_match(pred, gt, iou_threshold=0.5)
    assert len(matches) == 1 and matches[0][1] is True


def test_greedy_match_far_apart_is_fp():
    gt = [_box(10, 10, 40, 40)]
    pred = [_box(200, 200, 240, 240, score=0.8)]
    matches = _greedy_match(pred, gt, iou_threshold=0.5)
    assert len(matches) == 1 and matches[0][1] is False


def test_greedy_match_claims_best_iou_first():
    """Highest-score pred claims its best GT; second pred can't double-claim."""
    gt = [_box(10, 10, 50, 50)]
    pred = [
        _box(12, 12, 48, 48, score=0.95),   # IoU ~0.84 -> TP
        _box(14, 14, 46, 46, score=0.60),   # overlaps same GT -> already claimed -> FP
    ]
    matches = _greedy_match(pred, gt, iou_threshold=0.5)
    flags = [tp for _, tp in matches]
    assert flags == [True, False]


# ------------------------------------------------------------- dice / iou
def test_dice_iou_perfect_and_empty():
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 10:30, 10:30] = 1.0
    logits = torch.zeros(1, 1, 64, 64)
    logits[0, 0, 10:30, 10:30] = 20.0   # sigmoid -> ~1.0 exactly on target
    dice, iou = dice_iou_scores(logits, target)
    assert dice > 0.999 and iou > 0.999

    logits_empty = torch.full((1, 1, 64, 64), -20.0)  # sigmoid ~ 0 -> no predictions
    dice0, iou0 = dice_iou_scores(logits_empty, target)
    assert dice0 == 0.0 and iou0 == 0.0


def test_masks_to_boxes_matches_single_blob():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[20:40, 30:55] = 0.9
    boxes = masks_to_boxes(mask, threshold=0.5, min_area=10)
    assert len(boxes) == 1
    assert (boxes[0].y1, boxes[0].x1, boxes[0].y2, boxes[0].x2) == (20, 30, 40, 55)


# ------------------------------------- end-to-end on a synthetic checkpoint
@pytest.fixture(scope="module")
def synthetic_seg_env(tmp_path_factory):
    """Tiny DeepFish-style pair tree + untrained U-Net checkpoint."""
    from PIL import Image

    root = tmp_path_factory.mktemp("deepfish")
    size = 64
    rng = np.random.default_rng(0)
    for i in range(6):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        mask[10 + i:30 + i, 15:40] = 255
        Image.fromarray(img).save(root / "img" / f"f{i}.png")
        Image.fromarray(mask).save(root / "masks" / f"f{i}_mask.png")

    from src.segmentation.unet import SmallUNet

    device = torch.device("cpu")
    model = SmallUNet()
    ckpt_dir = root / "models" / "seg"
    ckpt_dir.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "image_size": size}, ckpt_dir / "best.pt")
    return root, ckpt_dir, device


def test_evaluate_segmentation_writes_metrics(synthetic_seg_env, tmp_path):
    root, ckpt_dir, device = synthetic_seg_env
    from src.segmentation.evaluate_seg import evaluate_segmentation

    metrics = evaluate_segmentation(
        models_dir=ckpt_dir.parent.parent,   # repo-style models/ dir
        results_dir=tmp_path / "results",
        device=device,
        data_dir=root,
        image_size=64,
        batch_size=2,
        min_box_area=5,
    )
    assert 0.0 <= metrics["iou"] <= 1.0
    assert 0.0 <= metrics["dice"] <= 1.0
    assert not np.isnan(metrics["pixel_ap"])
    assert not np.isnan(metrics["box_ap50"]) or metrics["n_pred_boxes"] == 0
    assert metrics["n_val_images"] >= 1
    saved = tmp_path / "results" / "seg" / "metrics.json"
    assert saved.exists()

    overlays = list((tmp_path / "results" / "seg").glob("overlay_*.png"))
    assert overlays, "qualitative overlays must be written"
