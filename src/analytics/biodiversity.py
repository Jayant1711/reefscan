"""Marine Biodiversity Score (MBS) — Phase 6 innovation #1.

Composition from species abundance counts ``n_i`` over ``S`` observed species
and ``N`` total detections:

    p_i        = n_i / N
    Shannon H' = -sum_i p_i * ln(p_i)                (uncertainty/diversity)
    Gini-Simpson 1-D = 1 - sum_i p_i^2               (dominance-inverse)
    Margalef d = (S - 1) / ln(N)                     (species richness)

    MBS = 100 * (w_h * H'/ln(S_ref)  +  w_s * (1-D)  +  w_m * min(d/d_ref, 1))

All three indices capture different facets: evenness, dominance, richness.
Defaults: w = (0.40, 0.30, 0.30), S_ref = 25 regional classes, d_ref = 3.
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np

DEFAULT_WEIGHTS = {"shannon": 0.40, "simpson": 0.30, "margalef": 0.30}


def _proportions(counts: Mapping[str, int]) -> np.ndarray:
    values = np.array(list(counts.values()), dtype=np.float64)
    if values.sum() <= 0:
        return values
    return values / values.sum()


def shannon_index(counts: Mapping[str, int]) -> float:
    """Shannon-Wiener index H' (nats)."""
    p = _proportions(counts)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def gini_simpson_index(counts: Mapping[str, int]) -> float:
    """Gini-Simpson diversity 1 - D in [0, 1)."""
    p = _proportions(counts)
    if p.size == 0:
        return 0.0
    return float(1.0 - (p**2).sum())


def margalef_richness(counts: Mapping[str, int]) -> float:
    """Margalef richness d = (S - 1) / ln(N)."""
    s = len(counts)
    n = sum(counts.values())
    if s <= 1 or n <= 1:
        return 0.0
    return float((s - 1) / np.log(n))


def marine_biodiversity_score(
    counts: Mapping[str, int],
    weights: Mapping[str, float] | None = None,
    shannon_ref_classes: int = 25,
    margalef_ref: float = 3.0,
) -> dict[str, float]:
    """Composite MBS in [0, 100] with component diagnostics."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if not counts or sum(counts.values()) == 0:
        return {"mbs": 0.0, "shannon": 0.0, "shannon_norm": 0.0,
                "gini_simpson": 0.0, "margalef": 0.0, "n_species": 0, "n_individuals": 0}

    h = shannon_index(counts)
    h_ref = np.log(max(2, min(shannon_ref_classes, max(len(counts), 2))))
    h_norm = h / h_ref

    gs = gini_simpson_index(counts)
    d = margalef_richness(counts)
    d_norm = min(d / margalef_ref, 1.0) if margalef_ref > 0 else 0.0

    mbs = 100.0 * (w["shannon"] * h_norm + w["simpson"] * gs + w["margalef"] * d_norm)
    return {
        "mbs": float(np.clip(mbs, 0, 100)),
        "shannon": h,
        "shannon_norm": h_norm,
        "gini_simpson": gs,
        "margalef": d,
        "n_species": len(counts),
        "n_individuals": int(sum(counts.values())),
    }


def rarity_report(counts: Mapping[str, int], rare_pct: float = 1.0) -> dict[str, list[str]]:
    """Species below a share threshold of total detections (Phase 6 #3)."""
    n = sum(counts.values())
    if n == 0:
        return {"rare": [], "common": [], "dominant": []}
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    rare = [k for k, v in counts.items() if 100.0 * v / n < rare_pct]
    return {
        "rare": rare,
        "common": [k for k, _ in ordered if k not in rare],
        "dominant": [ordered[0][0]] if ordered else [],
    }
