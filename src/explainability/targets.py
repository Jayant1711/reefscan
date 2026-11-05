"""Target-layer auto-detection for GradCAM-family methods per backbone."""

from __future__ import annotations

import torch.nn as nn


def find_target_layer(model: nn.Module, arch: str | None = None) -> nn.Module:
    """Return the final convolutional feature block for a known architecture.

    Falls back to the deepest Conv2d when the architecture is unknown, so new
    backbones work without code changes.

    Known mappings
    --------------
    - ResNet34            -> layer4[-1]                (512ch, 7x7 @224px)
    - EfficientNet-B0     -> features[-1]              (1280ch conv block)
    - MobileNetV3-Small   -> features[-1]              (576ch conv block)
    - ConvNeXt-Tiny       -> features[-1][-1]          (last CNBlock)
    """
    del arch  # architecture is inferred from structure, not the name

    if hasattr(model, "layer4"):                     # ResNet family
        return model.layer4[-1]
    if hasattr(model, "features"):                   # EfficientNet/MobileNet/ConvNeXt
        last = model.features[-1]
        if isinstance(last, nn.Sequential):          # ConvNeXt stage block
            return last[-1]
        return last
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if convs:
        return convs[-1]
    raise ValueError("Could not locate a convolutional target layer for CAM.")
