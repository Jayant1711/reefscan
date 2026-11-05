"""Image transforms with underwater-aware augmentation."""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn  # noqa: F401  (kept for parity with torchvision transforms v2)
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Underwater scenes: strong hue/saturation variability from depth, turbidity
# and lighting. Mild hue rotation simulates depth colour drift without
# destroying species-discriminative patterns.
UNDERWATER_JITTER = dict(brightness=0.30, contrast=0.30, saturation=0.30, hue=0.05)


def build_train_transform(cfg: Mapping[str, Any], image_size: int) -> T.Compose:
    aug = cfg.get("augmentation", {}) or {}
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(float(aug.get("horizontal_flip", 0.5))),
        T.RandomRotation(float(aug.get("rotation", 15))),
        T.ColorJitter(
            brightness=float(aug.get("brightness", 0.3)),
            contrast=float(aug.get("contrast", 0.3)),
            saturation=float(aug.get("saturation", 0.3)),
            hue=float(aug.get("hue", 0.05)),
        ),
        T.ToTensor(),
        T.RandomErasing(p=float(aug.get("random_erasing", 0.25)), value="random"),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
