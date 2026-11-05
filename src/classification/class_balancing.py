"""Class balancing: frequency analysis, inverse-sqrt class weights, weighted sampling."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import WeightedRandomSampler

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_class_images(root: str | Path) -> dict[str, int]:
    """Count images per class folder under an ImageFolder-style root."""
    root = Path(root)
    counts: dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
        if n:
            counts[class_dir.name] = n
    return counts


def class_weights(counts: dict[str, int], mode: str = "inverse_sqrt") -> np.ndarray:
    """Class weights ordered by sorted class name.

    - ``inverse``      : n_total / (C * n_c)  (sklearn 'balanced')
    - ``inverse_sqrt`` : gentler variant, robust to very rare classes
    """
    names = sorted(counts)
    freq = np.array([counts[n] for n in names], dtype=np.float64)
    total = freq.sum()
    if mode == "inverse":
        weights = total / (len(names) * freq)
    else:
        weights = np.sqrt(total / (len(names) * freq))
    return weights / weights.mean()  # normalise to mean 1


def sample_weights(dataset_targets: Sequence[int], counts: dict[str, int]) -> list[float]:
    """Per-sample weights for WeightedRandomSampler from class counts."""
    names = sorted(counts)
    freq = np.array([counts[n] for n in names], dtype=np.float64)
    inv = len(dataset_targets) / (len(names) * freq)
    per_class = {i: inv[i] for i in range(len(names))}
    return [per_class[t] for t in dataset_targets]


def make_weighted_sampler(dataset_targets: Sequence[int], counts: dict[str, int],
                          replacement: bool = True) -> WeightedRandomSampler:
    """Sampler that balances class exposure per epoch."""
    weights = sample_weights(dataset_targets, counts)
    return WeightedRandomSampler(torch_weights := weights, num_samples=len(weights), replacement=replacement)


def imbalance_ratio(counts: dict[str, int]) -> float:
    """max class size / min class size (1.0 = perfectly balanced)."""
    values = list(counts.values()) or [1]
    return float(max(values) / max(min(values), 1))
