"""Cached pipeline/model accessors for the dashboard (loaded once per session)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_resource(show_spinner="Loading ReefScan models (first time only) ...")
def get_pipeline(enable_cam: bool = False) -> Any:
    """Load the ReefScanPipeline once; falls back to demo checkpoints if needed."""
    from src.config import load_config
    from src.realtime.pipeline import ReefScanPipeline

    cfg = load_config()
    return ReefScanPipeline(cfg, enable_cam=enable_cam)


@st.cache_resource(show_spinner="Loading enhancement benchmark ...")
def get_enhancers():
    from src.enhancement.registry import build_enhancers
    from src.config import load_config

    cfg = load_config()
    return build_enhancers(cfg.enhancement.strategies, dict(cfg.enhancement.params))


@st.cache_data(show_spinner="Checking model status ...")
def model_status() -> dict[str, str]:
    """Which model sources are in use (best / demo / missing)."""
    from src.config import load_config

    cfg = load_config()
    status: dict[str, str] = {}
    for task in ("fish", "coral"):
        best_json = PROJECT_ROOT / cfg.paths.models_dir / task / "best_model.json"
        demo = PROJECT_ROOT / cfg.paths.demo_dir / f"{task}_demo.pt"
        if best_json.exists():
            status[task] = "trained"
        elif demo.exists():
            status[task] = "demo"
        else:
            status[task] = "missing"
    seg_ckpt = PROJECT_ROOT / cfg.paths.models_dir / "seg" / "best.pt"
    status["seg"] = "trained" if seg_ckpt.exists() else "missing"
    enh_report = PROJECT_ROOT / cfg.paths.results_dir / "enhancement" / "enhancement_report.json"
    status["enhancement"] = "auto-selected" if enh_report.exists() else "default (clahe)"
    return status


def demo_warning_banner() -> None:
    """Show a notice when demo (smoke-test) checkpoints are in use."""
    status = model_status()
    if status.get("fish") == "demo" or status.get("coral") == "demo":
        st.warning(
            "**Demo models active.** These lightweight checkpoints were trained on synthetic data by "
            "`run_smoke_test.py` — predictions are illustrative only. Train on real datasets for "
            "research-grade results (see `docs/TRAINING.md`).",
            icon="⚠️",
        )
    if status.get("fish") == "missing":
        st.error(
            "No fish model found. Run `python run_smoke_test.py` (instant demo) or "
            "`python train.py --task fish --model resnet34` (real training).",
            icon="🚨",
        )
