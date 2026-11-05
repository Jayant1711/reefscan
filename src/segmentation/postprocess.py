"""Post-processing: probability masks -> fish bounding boxes (connected components)."""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class BBox:
    """Axis-aligned detection box in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float          # mean mask probability inside the box
    area: int = 0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2, self.score]

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


def masks_to_boxes(
    prob_mask: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 150,
    max_boxes: int = 20,
    original_size: tuple[int, int] | None = None,
) -> list[BBox]:
    """Extract detection boxes from a probability mask.

    Parameters
    ----------
    prob_mask:
        float array [H, W] in [0, 1].
    threshold:
        binarisation threshold.
    min_area:
        discard components smaller than this many pixels (noise guard).
    max_boxes:
        keep the ``max_boxes`` largest components.
    original_size:
        optional (height, width) to rescale boxes back to source resolution.
    """
    binary = (prob_mask >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes: list[BBox] = []
    h, w = prob_mask.shape
    for label_id in range(1, n):  # 0 = background
        x, y, bw, bh, area = stats[label_id]
        if area < min_area:
            continue
        comp = (labels == label_id)
        score = float(prob_mask[comp].mean())
        boxes.append(BBox(int(x), int(y), int(x + bw), int(y + bh), score, int(area)))

    boxes.sort(key=lambda b: b.area, reverse=True)
    boxes = boxes[:max_boxes]

    if original_size is not None and (original_size[0], original_size[1]) != (h, w):
        oh, ow = original_size
        sy, sx = oh / h, ow / w
        boxes = [
            BBox(int(b.x1 * sx), int(b.y1 * sy), int(b.x2 * sx), int(b.y2 * sy), b.score, int(b.area * sx * sy))
            for b in boxes
        ]
    return boxes
