"""Phase 5 — Real-time analysis: unified image / video / webcam pipeline.

The torch-heavy pipeline is imported lazily so tracker/schema/annotate
consumers (analytics, tests) load without the DL stack.
"""

from typing import Any

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

_LAZY = {"ReefScanPipeline": ".pipeline"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
