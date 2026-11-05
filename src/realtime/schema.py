"""Dataclasses shared by the realtime pipeline, annotator and analytics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..segmentation.postprocess import BBox


@dataclass
class FishDetection:
    """One fish finding inside a frame."""

    box: BBox
    species: str
    confidence: float
    track_id: int | None = None

    @property
    def x1(self) -> int:
        return self.box.x1

    @property
    def y1(self) -> int:
        return self.box.y1

    @property
    def x2(self) -> int:
        return self.box.x2

    @property
    def y2(self) -> int:
        return self.box.y2


@dataclass
class CoralResult:
    """Coral branch output for one frame."""

    assessment: object  # SeverityAssessment (kept loose to avoid an import cycle)
    is_coral_scene: bool
    p_coral: float


@dataclass
class FrameAnalysis:
    """Complete analysis of one input image/frame."""

    detections: list[FishDetection]
    coral: CoralResult | None
    enhanced: np.ndarray
    heatmap: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)
    species_probs: dict[str, float] | None = None  # top-k of the best box (or frame)
    annotated: np.ndarray | None = None            # filled by analyze_file
    heatmap_path: str | None = None                # filled by analyze_file

    @property
    def dominant_species(self) -> str | None:
        if not self.detections:
            return None
        return max(self.detections, key=lambda d: d.confidence).species
