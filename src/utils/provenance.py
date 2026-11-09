"""Run provenance: git SHA, platform detection, GPU info, parameter counts.

Every training run stores this alongside its metrics so experiments from
Kaggle / Colab / Lightning / local can be compared honestly afterwards.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def git_sha(project_root: str | Path | None = None) -> str:
    """Short git SHA of the current checkout ('unknown' outside a repo)."""
    try:
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=10, check=False,
        )
        sha = out.stdout.strip()
        return sha if out.returncode == 0 and sha else "unknown"
    except Exception:  # pragma: no cover - git missing / permission issues
        return "unknown"


def detect_platform() -> str:
    """Identify the execution environment from well-known env variables."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle/working"):
        return "kaggle"
    if os.environ.get("COLAB_GPU") or os.path.exists("/content"):
        return "colab"
    if os.environ.get("LIGHTNING_CLOUD_PROJECT_ID") or os.environ.get("LIGHTNING_API_TOKEN"):
        return "lightning"
    return "local"


def gpu_name(device: "torch.device | None" = None) -> str:
    """Human-readable GPU name, or 'cpu'."""
    try:
        import torch

        if device is not None and device.type != "cuda":
            return "cpu"
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"
    except Exception:  # pragma: no cover - torch not installed
        return "unknown"


def count_params(model) -> dict[str, int]:
    """Total vs trainable parameter counts (exact, not rounded)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_manifest(
    *,
    task: str,
    model_name: str,
    strategy: str,
    dataset: str,
    config_snapshot: dict,
    params: dict,
    train_seconds: float,
    epochs_ran: int,
    final_metrics: dict,
    artifacts: dict[str, str] | None = None,
    project_root: str | Path | None = None,
) -> dict:
    """Assemble the canonical per-run manifest written to ``runs/<task>/...``."""
    return {
        "schema": "reefscan-run-manifest-v1",
        "recorded_at": utc_now_iso(),
        "git_sha": git_sha(project_root),
        "platform": detect_platform(),
        "gpu": gpu_name(),
        "task": task,
        "model": model_name,
        "strategy": strategy,
        "dataset": dataset,
        "config": config_snapshot,
        "params": params,
        "train_seconds": round(float(train_seconds), 1),
        "epochs_ran": int(epochs_ran),
        "final_metrics": {k: (round(v, 6) if isinstance(v, (int, float)) else v) for k, v in final_metrics.items()},
        "artifacts": artifacts or {},
    }
