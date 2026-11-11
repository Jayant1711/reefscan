"""Model factory: 4 torchvision backbones with classifier-head surgery."""

from __future__ import annotations

import logging
from typing import Sequence

import torch.nn as nn
from torchvision import models as tvm

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ["resnet34", "efficientnet_b0", "mobilenet_v3", "convnext_tiny"]


def _imagenet_weights(name: str):
    """Torchvision pretrained weights enum (with graceful fallback)."""
    mapping = {
        "resnet34": ("ResNet34_Weights", "IMAGENET1K_V1"),
        "efficientnet_b0": ("EfficientNet_B0_Weights", "IMAGENET1K_V1"),
        "mobilenet_v3": ("MobileNet_V3_Small_Weights", "IMAGENET1K_V1"),
        "convnext_tiny": ("ConvNeXt_Tiny_Weights", "IMAGENET1K_V1"),
    }
    enum_name, weights_name = mapping[name]
    try:
        return getattr(tvm, enum_name).IMAGENET1K_V1
    except AttributeError:  # very old torchvision
        return True


def _replace_head(model: nn.Module, name: str, num_classes: int, dropout: float) -> nn.Module:
    if name == "resnet34":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "mobilenet_v3":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "convnext_tiny":
        in_features = model.classifier[-1].in_features if isinstance(model.classifier[-1], nn.Linear) else model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:  # pragma: no cover - guarded by build_model
        raise ValueError(name)
    return model


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2,
                freeze_backbone: bool = False) -> nn.Module:
    """Instantiate a backbone and graft a fresh classification head on it.

    Parameters
    ----------
    name:
        One of :data:`SUPPORTED_MODELS`.
    num_classes:
        Number of output classes.
    pretrained:
        Use ImageNet-initialised weights (transfer learning).
    dropout:
        Dropout probability right before the final linear layer.
    freeze_backbone:
        Linear-probe mode: freeze everything except the new head.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Supported: {SUPPORTED_MODELS}")

    weights = _imagenet_weights(name) if pretrained else None
    if name == "resnet34":
        model = tvm.resnet34(weights=weights)
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=weights)
    elif name == "mobilenet_v3":
        model = tvm.mobilenet_v3_small(weights=weights)
    else:  # convnext_tiny
        model = tvm.convnext_tiny(weights=weights)

    model = _replace_head(model, name, num_classes, dropout)

    if freeze_backbone:
        head_markers = ("fc", "classifier")
        for param_name, param in model.named_parameters():
            if not any(marker in param_name for marker in head_markers):
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Built %s (classes=%d, trainable=%.2fM / %.2fM params)",
                name, num_classes, trainable / 1e6, total / 1e6)
    return model


def param_groups(model: nn.Module, lr: float, head_lr_multiplier: float = 10.0) -> list[dict]:
    """Discriminative learning rates: fresh head trains faster than the backbone."""
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("fc" in n or "classifier" in n)]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not ("fc" in n or "classifier" in n)]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr})
    if head_params:
        groups.append({"params": head_params, "lr": lr * head_lr_multiplier})
    return groups


# Final stage per backbone (verified param-name prefixes on torchvision).
# ConvNeXt has NO top-level `stages` — stages live at features.1/3/5/7.
_LAST_STAGE = {
    "resnet34": "layer4",
    "efficientnet_b0": "features.7",
    "mobilenet_v3": "features.12",
    "convnext_tiny": "features.7",
}


def unfreeze_stage(model: nn.Module, name: str) -> None:
    """Un-freeze the final stage of a backbone (partial fine-tuning).

    The head stays trainable, the earlier backbone blocks stay frozen.
    Must be called *after* ``freeze_backbone`` has been applied.
    """
    if name not in _LAST_STAGE:
        raise ValueError(f"Unknown last stage for '{name}'. Supported: {sorted(_LAST_STAGE)}")
    prefix = _LAST_STAGE[name]
    unfrozen = 0
    for param_name, param in model.named_parameters():
        if "fc" in param_name or "classifier" in param_name:
            param.requires_grad = True  # head always trains
        elif param_name.startswith(prefix):
            param.requires_grad = True
            unfrozen += param.numel()
    logger.info("Unfroze last stage '%s' of %s (%.2fM params now trainable).", prefix, name, unfrozen / 1e6)


def partial_finetune_groups(model: nn.Module, name: str, lr: float,
                            head_lr_multiplier: float = 10.0,
                            stage_lr_multiplier: float = 0.1) -> list[dict]:
    """Optimizer groups for partial fine-tuning, **including dormant frozen params**.

    The model is built with ``freeze_backbone=True``; frozen backbone params are
    still put into the optimizer at the stage LR so that calling
    :func:`unfreeze_stage` mid-training requires no optimizer rebuild — dormant
    params simply never receive gradients (and are skipped by AdamW) until
    ``requires_grad`` flips to True.
    """
    prefix = _LAST_STAGE.get(name, "")
    head_params, stage_params, frozen_params = [], [], []
    for param_name, param in model.named_parameters():
        if "fc" in param_name or "classifier" in param_name:
            head_params.append(param)
        elif prefix and param_name.startswith(prefix):
            stage_params.append(param)
        else:
            frozen_params.append(param)
    groups = []
    if frozen_params:
        groups.append({"params": frozen_params, "lr": lr * stage_lr_multiplier})
    if stage_params:
        groups.append({"params": stage_params, "lr": lr * stage_lr_multiplier})
    if head_params:
        groups.append({"params": head_params, "lr": lr * head_lr_multiplier})
    return groups


def class_names_from_model(ckpt: dict) -> Sequence[str]:
    """Class names stored inside a checkpoint dict."""
    return ckpt.get("class_names", [])
