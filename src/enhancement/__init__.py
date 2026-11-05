"""Phase 1 — Underwater image enhancement (8 strategies + no-reference metrics + auto-selection)."""

from .base import Enhancer
from .registry import ENHANCER_REGISTRY, build_enhancers, get_enhancer
from .metrics import psnr, ssim, uciqe, uiqm, evaluate_enhancement
from .selector import EnhancementSelector

__all__ = [
    "Enhancer",
    "ENHANCER_REGISTRY",
    "build_enhancers",
    "get_enhancer",
    "psnr",
    "ssim",
    "uciqe",
    "uiqm",
    "evaluate_enhancement",
    "EnhancementSelector",
]
