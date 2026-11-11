"""Final segmentation evaluation: IoU, Dice, pixel AP, box AP@0.5, overlays.

Consumes ``models/seg/best.pt`` (written by :class:`SegmentationTrainer`) and
produces ``results/seg/metrics.json`` plus overlay PNGs — the numbers recorded
in ``my_requested_thing.md`` come from here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..utils import ensure_dir, save_json
from .engine_seg import dice_iou_scores, make_seg_loaders
from .postprocess import masks_to_boxes
from .unet import SmallUNet

logger = logging.getLogger(__name__)

PIXEL_STRIDE = 2          # subsample pixels for the (large) pixel-AP arrays
OVERLAY_LIMIT = 4         # how many qualitative overlay PNGs to save


@torch.no_grad()
def evaluate_segmentation(
    models_dir: str | Path,
    results_dir: str | Path,
    device: torch.device,
    data_dir: str | Path,
    image_size: int = 256,
    batch_size: int = 8,
    min_box_area: int = 150,
    iou_threshold: float = 0.5,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the trained U-Net on the validation split and persist metrics."""
    from sklearn.metrics import average_precision_score

    ckpt_path = Path(models_dir) / "seg" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No segmentation checkpoint at {ckpt_path} — train seg first.")

    bundle = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = SmallUNet().to(device)
    model.load_state_dict(bundle["model"])
    model.eval()
    # Prefer the size the model was trained at (checkpoint stores it).
    image_size = int(bundle.get("image_size", image_size))

    _, val_loader = make_seg_loaders(data_dir, image_size=image_size, batch_size=batch_size)
    if val_loader is None:
        raise FileNotFoundError(f"No validation pairs under {data_dir}")

    dices, ious = [], []
    pixel_probs: list[np.ndarray] = []
    pixel_targets: list[np.ndarray] = []
    all_pred_boxes: list[tuple[float, bool]] = []   # (score, is_true_positive)
    n_gt_total, n_pred_total = 0, 0
    overlays_saved = 0

    out_dir = ensure_dir(Path(results_dir) / "seg")
    for batch_idx, (imgs, masks) in enumerate(val_loader):
        imgs = imgs.to(device)
        masks_dev = masks.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)

        d, i = dice_iou_scores(logits, masks_dev, threshold=threshold)
        dices.append(d)
        ious.append(i)

        # Pixel-level AP (subsampled for memory).
        p = probs[:, 0].cpu().numpy()[:, ::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel()
        t = masks[:, 0].numpy()[:, ::PIXEL_STRIDE, ::PIXEL_STRIDE].ravel()
        pixel_probs.append(p)
        pixel_targets.append(t)

        # Box-level AP@0.5 via greedy IoU matching per image.
        for b in range(imgs.size(0)):
            prob_mask = probs[b, 0].cpu().numpy()
            gt_mask = masks[b, 0].numpy()
            gt_boxes = masks_to_boxes(gt_mask, threshold=0.5, min_area=1)
            pred_boxes = masks_to_boxes(prob_mask, threshold=threshold, min_area=min_box_area)
            n_gt_total += len(gt_boxes)
            n_pred_total += len(pred_boxes)

            matched = _greedy_match(pred_boxes, gt_boxes, iou_threshold)
            for pred_box, is_tp in matched:
                all_pred_boxes.append((float(pred_box.score), bool(is_tp)))

            if overlays_saved < OVERLAY_LIMIT and batch_idx == 0:
                _save_overlay(imgs[b], prob_mask, gt_mask, pred_boxes, gt_boxes,
                              out_dir / f"overlay_{overlays_saved}.png")
                overlays_saved += 1

    y_prob_pix = np.concatenate(pixel_probs)
    y_true_pix = np.concatenate(pixel_targets)
    pixel_ap = float(average_precision_score(y_true_pix.astype(int), y_prob_pix)) if y_true_pix.any() else float("nan")

    # Global box AP@0.5 from (score, tp) pairs — a VOC-style single-point AP.
    box_ap50 = float("nan")
    if all_pred_boxes and n_gt_total > 0:
        scores = np.array([s for s, _ in all_pred_boxes])
        tps = np.array([1 if tp else 0 for _, tp in all_pred_boxes])
        if len(np.unique(scores)) > 1:
            box_ap50 = float(average_precision_score(tps, scores))
        else:
            box_ap50 = float(tps.mean())

    metrics = {
        "n_val_images": int(len(val_loader.dataset)),
        "iou": float(np.mean(ious)),
        "dice": float(np.mean(dices)),
        "pixel_ap": pixel_ap,
        "box_ap50": box_ap50,
        "n_gt_boxes": int(n_gt_total),
        "n_pred_boxes": int(n_pred_total),
        "iou_threshold": iou_threshold,
        "min_box_area": min_box_area,
        "threshold": threshold,
    }
    save_json(metrics, out_dir / "metrics.json")
    logger.info("Seg eval: IoU=%.4f Dice=%.4f pixelAP=%.4f boxAP@0.5=%.4f",
                metrics["iou"], metrics["dice"], pixel_ap, box_ap50)
    return metrics


def _greedy_match(pred_boxes, gt_boxes, iou_threshold: float) -> list[tuple[object, bool]]:
    """Sort predictions by score, greedily match to the best unclaimed GT."""
    matches: list[tuple[object, bool]] = []
    claimed: set[int] = set()
    for pred in sorted(pred_boxes, key=lambda b: b.score, reverse=True):
        best_iou, best_gt = 0.0, -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in claimed:
                continue
            iou = pred.iou(gt)
            if iou > best_iou:
                best_iou, best_gt = iou, gt_idx
        if best_gt >= 0 and best_iou >= iou_threshold:
            claimed.add(best_gt)
            matches.append((pred, True))
        else:
            matches.append((pred, False))
    return matches


def _save_overlay(img_tensor: torch.Tensor, prob_mask: np.ndarray, gt_mask: np.ndarray,
                  pred_boxes, gt_boxes, out_path: Path) -> None:
    """Qualitative side-by-side: image | GT boxes | predicted boxes."""
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = (img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    def draw(boxes, color):
        canvas = img.copy()
        for box in boxes:
            cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        return canvas

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("image")
    axes[1].imshow(draw(gt_boxes, (0, 220, 0)))
    axes[1].set_title(f"GT boxes ({len(gt_boxes)})")
    axes[2].imshow(draw(pred_boxes, (220, 120, 0)))
    axes[2].set_title(f"predicted ({len(pred_boxes)})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
