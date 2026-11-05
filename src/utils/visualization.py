"""Matplotlib visualisation helpers (training curves, confusion matrices, grids)."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe on servers / Colab
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(history: Sequence[Mapping[str, float]], out_path: str | Path) -> Path:
    """Plot loss and F1/accuracy curves from an epoch-history list of dicts."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], "o-", label="train loss")
    if any("val_loss" in h for h in history):
        axes[0].plot(epochs, [h.get("val_loss", float("nan")) for h in history], "s-", label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    metric_keys = [k for k in history[0].keys() if k.startswith("val_") and k != "val_loss"]
    for key in metric_keys:
        axes[1].plot(epochs, [h[key] for h in history], "o-", label=key.replace("val_", "val "))
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    return _save(fig, out_path)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> Path:
    """Render a labelled confusion matrix heatmap."""
    matrix = np.asarray(matrix, dtype=np.float64)
    shown = matrix.copy()
    fmt = "d"
    if normalize and matrix.sum() > 0:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        shown = matrix / row_sums
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(class_names) + 2), max(5, 0.6 * len(class_names) + 1)))
    im = ax.imshow(shown, cmap="Blues", vmin=0.0 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    labels = [n[:22] for n in class_names]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    thresh = shown.max() / 2.0 if shown.size else 0.5
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            ax.text(
                j, i, format(shown[i, j], fmt),
                ha="center", va="center", fontsize=7,
                color="white" if shown[i, j] > thresh else "black",
            )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    return _save(fig, out_path)


def image_grid(
    images: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    cols: int | None = None,
    figsize_per_cell: float = 3.0,
) -> plt.Figure | Path:
    """Render an n-up grid of RGB images (optionally saving to ``out_path``)."""
    n = len(images)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_cell * cols, figsize_per_cell * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for idx, img in enumerate(images):
        axes[idx].imshow(np.clip(img, 0, 1) if img.dtype != np.uint8 else img)
        if titles is not None and idx < len(titles):
            axes[idx].set_title(titles[idx], fontsize=9)
    if out_path is None:
        return fig
    return _save(fig, out_path)


def bar_chart(
    values: Mapping[str, float],
    out_path: str | Path,
    title: str = "",
    ylabel: str = "",
    horizontal: bool = False,
    color: str = "#1f77b4",
) -> Path:
    """Simple labelled bar chart."""
    names = list(values.keys())
    vals = list(values.values())
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(names) + 3), 4.5))
    if horizontal:
        ax.barh(names, vals, color=color)
        ax.set_xlabel(ylabel)
        ax.invert_yaxis()
    else:
        ax.bar(names, vals, color=color)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x" if horizontal else "y")
    return _save(fig, out_path)


def severity_gauge(index: float, out_path: str | Path, title: str = "Bleaching Severity Index") -> Path:
    """Semi-circular 0-100 gauge with traffic-light colouring."""
    index = float(np.clip(index, 0, 100))
    fig, ax = plt.subplots(figsize=(5.2, 3.4), subplot_kw={"aspect": "equal"})
    bands = [(0, 15, "#2ecc71"), (15, 35, "#f1c40f"), (35, 60, "#e67e22"), (60, 85, "#e74c3c"), (85, 100, "#7f8c8d")]
    for lo, hi, color in bands:
        theta = np.linspace(np.pi * (1 - lo / 100), np.pi * (1 - hi / 100), 24)
        ax.plot(np.cos(theta), np.sin(theta), lw=16, color=color, solid_capstyle="butt")
    needle = np.pi * (1 - index / 100)
    ax.annotate("", xy=(0.82 * np.cos(needle), 0.82 * np.sin(needle)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2))
    ax.text(0, -0.18, f"{index:.1f}", ha="center", fontsize=22, fontweight="bold")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.2)
    ax.axis("off")
    ax.set_title(title, fontsize=11)
    return _save(fig, out_path)
