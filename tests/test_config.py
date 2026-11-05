"""Config system tests."""

from __future__ import annotations

from src.config import DotDict, load_config


def test_dot_access_and_defaults():
    cfg = load_config()
    assert cfg.project.name == "ReefScan AI"
    assert cfg.fish.image_size == 224
    assert len(cfg.enhancement.strategies) == 8
    assert cfg.coral.severity.thresholds == [15, 35, 60, 85]


def test_overrides_coercion():
    cfg = load_config(overrides=["fish.epochs=3", "fish.pretrained=false",
                                 "coral.severity.fusion_lambda=0.75", "xai.scorecam_max_channels=16"])
    assert cfg.fish.epochs == 3
    assert cfg.fish.pretrained is False
    assert abs(cfg.coral.severity.fusion_lambda - 0.75) < 1e-9
    assert cfg.xai.scorecam_max_channels == 16


def test_nested_set_and_get():
    d = DotDict({"a": {"b": {"c": 1}}})
    assert d.get_nested("a.b.c") == 1
    assert d.get_nested("a.b.x", default=7) == 7
    d.set_nested("a.b.c", 99)
    d.set_nested("x.y.z", 5)
    assert d.a.b.c == 99
    assert d.get_nested("x.y.z") == 5
