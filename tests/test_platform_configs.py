"""Platform-profile pins: every overlay loads, and the gpu workstation profile
(HANDOFF.md) carries the keys the local-GPU run depends on.

Note: importlib-free — the local Windows WDAC policy intermittently blocks
pyarrow/sklearn/torch DLLs, so these tests only use yaml + source inspection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import load_config, merge_config_file

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ["kaggle", "colab", "lightning", "gpu", "local"]


def _with_profile(name: str):
    cfg = load_config()
    merge_config_file(cfg, ROOT / "configs" / "platform" / f"{name}.yaml")
    return cfg


@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads_and_is_wellformed(name: str):
    cfg = _with_profile(name)
    for key in ("data_root", "models_dir", "results_dir", "reports_dir", "demo_dir"):
        assert cfg.paths[key], f"{name}: paths.{key} empty"
    # Training knobs the CLIs read unconditionally must survive the merge.
    assert cfg.fish.image_size == 224
    assert cfg.coral.severity.thresholds == [15, 35, 60, 85]
    assert cfg.seg.threshold == 0.5


def test_gpu_profile_pins_throughput_keys():
    cfg = _with_profile("gpu")
    assert cfg.device.num_workers == 8
    assert cfg.fish.batch_size == 64
    assert cfg.coral.batch_size == 64
    assert cfg.seg.batch_size == 16
    # GPU box keeps repo-relative paths (datasets download inside the clone).
    assert not str(cfg.paths.data_root).startswith("/")


def test_cloud_profiles_pin_ephemeral_data_roots():
    assert _with_profile("kaggle").paths.data_root.startswith("/kaggle/tmp")
    assert _with_profile("colab").paths.data_root.startswith("/content")


def test_local_profile_conservative_workers():
    assert _with_profile("local").device.num_workers == 0


def test_gpu_choice_present_in_all_platform_cli_sources():
    """'gpu' must be an accepted --platform choice in every CLI that takes one."""
    sources = [
        ROOT / "train.py",
        ROOT / "evaluate.py",
        ROOT / "download_data.py",
        ROOT / "clean_data.py",
        ROOT / "scripts" / "preflight.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert 'choices=["kaggle", "colab", "lightning", "gpu", "local"]' in text, \
            f"{path.name}: --platform choices missing 'gpu'"


def test_handoff_doc_exists_and_covers_matrix():
    text = (ROOT / "HANDOFF.md").read_text(encoding="utf-8")
    for token in ("E1", "E6", "--platform gpu", "Step 9", "results/local-gpu", "preflight"):
        assert token in text, f"HANDOFF.md missing token: {token}"
