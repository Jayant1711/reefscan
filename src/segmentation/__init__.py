"""Phase 3.5 — Fish segmentation (DeepFish masks) for detection boxes in real-time view."""

from .unet import SmallUNet
from .postprocess import BBox, masks_to_boxes
from .engine_seg import SegmentationTrainer

__all__ = ["SmallUNet", "BBox", "masks_to_boxes", "SegmentationTrainer"]
