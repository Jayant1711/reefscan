"""Phase 5 — Real-time analysis: unified image / video / webcam pipeline."""

from .pipeline import ReefScanPipeline
from .schema import CoralResult, FrameAnalysis, FishDetection
from .tracker import CentroidTracker
from .annotate import annotate_frame

__all__ = [
    "ReefScanPipeline",
    "FrameAnalysis",
    "FishDetection",
    "CoralResult",
    "CentroidTracker",
    "annotate_frame",
]
