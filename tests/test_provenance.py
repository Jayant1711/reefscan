"""Provenance + strategy tests: manifests, param counting, freeze/unfreeze."""

from __future__ import annotations

import torch.nn as nn

from src.classification.engine import STRATEGIES, TrainConfig
from src.classification.models import (
    SUPPORTED_MODELS,
    build_model,
    partial_finetune_groups,
    param_groups,
    unfreeze_stage,
)
from src.utils.provenance import build_manifest, count_params, detect_platform, git_sha


# ---------------------------------------------------------------- provenance
def test_count_params_exact():
    model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))  # 55 + 12 = 67
    counts = count_params(model)
    assert counts == {"params_total": 67, "params_trainable": 67}
    model[1].weight.requires_grad = False   # freezes 2x5 = 10 params
    counts = count_params(model)
    assert counts["params_trainable"] == 57 and counts["params_total"] == 67


def test_detect_platform_returns_known_value():
    assert detect_platform() in {"kaggle", "colab", "lightning", "local"}


def test_git_sha_is_string():
    sha = git_sha()
    assert isinstance(sha, str) and sha


def test_build_manifest_fields():
    manifest = build_manifest(
        task="coral",
        model_name="efficientnet_b0",
        strategy="feature_extraction",
        dataset="datasets/coral/imagefolders",
        config_snapshot={"epochs": 10},
        params={"params_total": 100, "params_trainable": 50},
        train_seconds=61.4,
        epochs_ran=3,
        final_metrics={"accuracy": 0.91234, "macro_ap": 0.85, "weird": [1, 2]},
    )
    assert manifest["schema"] == "reefscan-run-manifest-v2"
    assert manifest["task"] == "coral" and manifest["strategy"] == "feature_extraction"
    assert manifest["train_seconds"] == 61.4 and manifest["epochs_ran"] == 3
    assert manifest["final_metrics"]["accuracy"] == 0.91234
    assert manifest["final_metrics"]["weird"] == [1, 2]      # non-scalar passed through
    assert manifest["gpu"] in {"cpu", "unknown"} or "GPU" in manifest["gpu"]
    for key in ("git_sha", "platform", "recorded_at"):
        assert manifest[key]


# ------------------------------------------------------------------ strategy
def test_trainconfig_accepts_strategy_keys():
    cfg = TrainConfig.from_cfg({"epochs": 2, "strategy": "partial_finetune", "freeze_epochs": 2})
    assert cfg.strategy == "partial_finetune" and cfg.freeze_epochs == 2


def test_trainconfig_rejects_unknown_strategy():
    try:
        TrainConfig.from_cfg({"strategy": "nope"})
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown strategy must raise ValueError")


def test_strategies_constant_matches_cli():
    assert set(STRATEGIES) == {"feature_extraction", "finetune", "partial_finetune"}


def test_feature_extraction_freezes_backbone():
    model = build_model("resnet34", num_classes=3, freeze_backbone=True, pretrained=False)
    head_trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert head_trainable, "head must stay trainable"
    assert all("fc" in n for n in head_trainable), "only head params train in FE mode"
    counts = count_params(model)
    assert counts["params_trainable"] < counts["params_total"]


def test_unfreeze_stage_resnet_only_last_block():
    model = build_model("resnet34", num_classes=3, freeze_backbone=True, pretrained=False)
    before = count_params(model)["params_trainable"]
    unfreeze_stage(model, "resnet34")
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith("layer4") for n in trainable)
    assert not any(n.startswith("layer3") for n in trainable), "earlier blocks stay frozen"
    after = count_params(model)["params_trainable"]
    assert after > before


def test_unfreeze_stage_all_backbones():
    for name in SUPPORTED_MODELS:
        model = build_model(name, num_classes=2, freeze_backbone=True, pretrained=False)
        unfreeze_stage(model, name)  # must not raise
        assert any(p.requires_grad for n, p in model.named_parameters() if n.startswith(
            {"resnet34": "layer4", "efficientnet_b0": "features.7",
             "mobilenet_v3": "features.12", "convnext_tiny": "features.7"}[name]
        ))


def test_partial_finetune_groups_include_dormant_params():
    """Frozen backbone params stay in the optimizer so mid-training unfreeze needs no rebuild."""
    model = build_model("efficientnet_b0", num_classes=2, freeze_backbone=True, pretrained=False)
    groups = partial_finetune_groups(model, "efficientnet_b0", lr=1e-4)
    grouped = sum(len(g["params"]) for g in groups)
    total = len(list(model.parameters()))
    assert grouped == total, "partial groups must cover every parameter (incl. dormant)"
    lrs = [g["lr"] for g in groups]
    assert max(lrs) == 1e-3  # head at 10x


# NOTE: pretrained=False everywhere — structure tests, no 230MB weight download
# (keeps the platform pytest gate fast and network-free).


def test_param_groups_excludes_frozen():
    model = build_model("resnet34", num_classes=2, freeze_backbone=True, pretrained=False)
    groups = param_groups(model, lr=1e-4)
    n_optim = sum(len(g["params"]) for g in groups)
    assert n_optim == sum(1 for p in model.parameters() if p.requires_grad)
