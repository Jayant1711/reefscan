"""Frame annotation: detection boxes, species labels, coral status banner, CAM overlay."""

from __future__ import annotations

import cv2
import numpy as np

from ..coral_analysis.severity import SeverityAssessment
from ..explainability.visualize import overlay_cam_on_image
from .schema import FishDetection

# Traffic-light colours in BGR
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "healthy": (113, 204, 46),
    "mild": (15, 196, 241),
    "moderate": (34, 126, 230),
    "severe": (60, 76, 231),
    "dead": (140, 140, 140),
}
BOX_COLOR = (90, 220, 60)
TEXT_COLOR = (255, 255, 255)


def annotate_frame(
    frame_bgr: np.ndarray,
    detections: list[FishDetection],
    coral: SeverityAssessment | None = None,
    heatmap: np.ndarray | None = None,
    fps: float | None = None,
    frame_idx: int | None = None,
    heatmap_alpha: float = 0.35,
) -> np.ndarray:
    """Render an analysis frame ready for video writing or display."""
    out = frame_bgr.copy()
    if heatmap is not None:
        out = overlay_cam_on_image(out, heatmap, alpha=heatmap_alpha)

    for det in detections:
        color = BOX_COLOR if det.confidence >= 0.6 else (80, 170, 250)
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.species} {det.confidence:.0%}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = det.y1 - 6 if det.y1 - 6 > th else det.y2 + th + 6
        cv2.rectangle(out, (det.x1, ty - th - 4), (det.x1 + tw + 4, ty + 4), color, -1)
        cv2.putText(out, label, (det.x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)

    if coral is not None:
        color = CLASS_COLORS_BGR.get(coral.class_name, (200, 200, 200))
        banner = f"Coral: {coral.label} | BSI {coral.index:.0f}/100"
        cv2.rectangle(out, (10, 10), (10 + 8 * len(banner) + 12, 42), color, -1)
        cv2.putText(out, banner, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)
        # Severity progress bar
        bar_w = int(180 * np.clip(coral.index, 0, 100) / 100)
        cv2.rectangle(out, (10, 48), (198, 58), (60, 60, 60), -1)
        cv2.rectangle(out, (10, 48), (10 + bar_w, 58), color, -1)

    hud = []
    if fps is not None:
        hud.append(f"FPS {fps:.1f}")
    if frame_idx is not None:
        hud.append(f"frame {frame_idx}")
    if detections:
        top = max(detections, key=lambda d: d.confidence)
        hud.append(f"top: {top.species} ({top.confidence:.0%})")
    if hud:
        text = " | ".join(hud)
        cv2.putText(out, text, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)
    return out
