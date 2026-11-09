"""Evaluation pipelines: per-model metrics, confusion matrices, model leaderboard."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ..utils import ensure_dir, load_json, save_json
from ..utils.visualization import plot_confusion_matrix, plot_training_curves
from .datasets import build_dataloaders
from .engine import TrainConfig, Trainer, evaluate_model_loader
from .models import build_model

logger = logging.getLogger(__name__)


def _load_checkpoint_model(model_dir: Path, num_classes: int, device: torch.device):
    ckpt = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], ckpt.get("num_classes", num_classes), pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def run_model_evaluation(
    task_name: str,
    model_name: str,
    data_dir: str | Path,
    models_dir: str | Path,
    results_dir: str | Path,
    class_names: Sequence[str],
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Evaluate one trained checkpoint on the test (or val) split and persist artefacts."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = ensure_dir(Path(models_dir) / task_name / model_name)
    out_dir = ensure_dir(Path(results_dir) / task_name / model_name)

    bundle = build_dataloaders(data_dir, image_size=image_size, batch_size=batch_size,
                               num_workers=num_workers, balance_train=False)
    eval_loader = bundle.test or bundle.val
    if eval_loader is None:
        raise FileNotFoundError(f"No test/validation split under {data_dir}")

    model, ckpt = _load_checkpoint_model(model_dir, bundle.num_classes, device)
    metrics = evaluate_model_loader(model, eval_loader, device, bundle.num_classes)

    # Persist headline metrics (drop bulky per-sample arrays).
    headline = {k: v for k, v in metrics.items() if k not in {"y_true", "y_pred", "y_prob"}}
    save_json(headline, out_dir / "metrics.json")

    # Per-class CSV (now includes Average Precision).
    ap = metrics.get("per_class_ap") or []
    per_class = pd.DataFrame({
        "class": bundle.class_names,
        "precision": metrics["per_class_precision"],
        "recall": metrics["per_class_recall"],
        "f1": metrics["per_class_f1"],
        "ap": [ap[i] if i < len(ap) else float("nan") for i in range(bundle.num_classes)],
        "support": [int(np.sum(np.array(metrics["y_true"]) == i)) for i in range(bundle.num_classes)],
    })
    per_class.to_csv(out_dir / "per_class_report.csv", index=False)

    # Confusion matrix + curves.
    from sklearn.metrics import confusion_matrix, precision_recall_curve

    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(range(bundle.num_classes)))
    plot_confusion_matrix(cm, bundle.class_names, out_dir / "confusion_matrix.png",
                          title=f"{model_name} confusion matrix")

    # One-vs-rest precision-recall curves with per-class AP in the legend.
    if metrics.get("y_prob") and bundle.num_classes > 1:
        _plot_pr_curves(np.array(metrics["y_true"]), np.array(metrics["y_prob"]),
                        bundle.class_names, out_dir / "pr_curves.png", model_name)

    history_path = model_dir / "history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path).to_dict(orient="records")
        plot_training_curves(history, out_dir / "training_curves.png")

    logger.info("%s/%s: acc=%.4f wF1=%.4f top5=%.4f mAP=%.4f (%d samples)",
                task_name, model_name, metrics["accuracy"], metrics["weighted_f1"],
                metrics["top5_accuracy"], metrics.get("macro_ap", float("nan")), metrics["n"])
    return metrics


def _plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str],
                    out_path: Path, model_name: str) -> None:
    """One-vs-rest precision-recall curves, one line per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(class_names):
        binary = (y_true == c).astype(int)
        if binary.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary, y_prob[:, c])
        ap = average_precision_score(binary, y_prob[:, c])
        ax.plot(recall, precision, lw=1.8, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{model_name} — one-vs-rest PR curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_leaderboard(
    task_name: str,
    model_names: Sequence[str],
    results_dir: str | Path,
    models_dir: str | Path,
    metric_for_best: str = "weighted_f1",
) -> tuple[pd.DataFrame, str]:
    """Aggregate metrics.json across models -> leaderboard.csv + best model name."""
    rows = []
    for name in model_names:
        metrics_path = Path(results_dir) / task_name / name / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Missing metrics for %s/%s — skipping.", task_name, name)
            continue
        m = load_json(metrics_path)
        # Strategy + params + train time from best_meta.json / history.csv.
        meta = load_json(Path(models_dir) / task_name / name / "best_meta.json") or {}
        params = meta.get("params") or {}
        strategy = meta.get("strategy", "finetune")
        params_m = round(params.get("params_total", 0) / 1e6, 2)
        trainable_m = round(params.get("params_trainable", 0) / 1e6, 2)
        train_seconds = _total_train_seconds(Path(models_dir) / task_name / name / "history.csv")
        rows.append({
            "model": name,
            "strategy": strategy,
            "params_m": params_m,
            "params_trainable_m": trainable_m,
            "train_seconds": train_seconds,
            "accuracy": m.get("accuracy"),
            "top5_accuracy": m.get("top5_accuracy"),
            "macro_ap": m.get("macro_ap"),
            "precision_macro": m.get("precision_macro"),
            "recall_macro": m.get("recall_macro"),
            "macro_f1": m.get("macro_f1"),
            "weighted_f1": m.get("weighted_f1"),
            "throughput_fps": m.get("throughput_fps"),
            "n_test": m.get("n"),
        })
    if not rows:
        raise FileNotFoundError(f"No evaluation results under {results_dir}/{task_name}")

    df = pd.DataFrame(rows).sort_values(metric_for_best, ascending=False).reset_index(drop=True)
    out_dir = ensure_dir(Path(results_dir) / task_name)
    df.to_csv(out_dir / "model_leaderboard.csv", index=False)
    best = str(df.iloc[0]["model"])

    save_json(
        {"task": task_name, "best_model": best, "metric_for_best": metric_for_best,
         "leaderboard": df.to_dict(orient="records")},
        Path(models_dir) / task_name / "best_model.json",
    )
    logger.info("Best %s model: %s (%s)", task_name, best, metric_for_best)
    return df, best


def _total_train_seconds(history_csv: Path) -> float:
    """Sum of per-epoch training seconds from history.csv (0.0 if missing)."""
    if not history_csv.exists():
        return 0.0
    try:
        history = pd.read_csv(history_csv)
        if "seconds" in history.columns:
            return float(history["seconds"].fillna(0).sum())
    except Exception:  # pragma: no cover - malformed history
        pass
    return 0.0
