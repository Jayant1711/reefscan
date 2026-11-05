"""Chart assets embedded into PDF reports and the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils import ensure_dir
from ..utils.visualization import _save

CLASS_ORDER = ["healthy", "mild", "moderate", "severe", "dead"]
CLASS_COLORS = {"healthy": "#2ecc71", "mild": "#f1c40f", "moderate": "#e67e22",
                "severe": "#e74c3c", "dead": "#7f8c8d"}


def species_distribution_chart(counts: Mapping[str, int], out_path: str | Path) -> Path:
    """Horizontal bar chart of species abundances."""
    names = list(counts.keys()) or ["(no fish detected)"]
    values = [counts[k] for k in names] if counts else [0]
    fig, ax = plt.subplots(figsize=(6.4, max(2.2, 0.45 * len(names) + 1.2)))
    ax.barh(names, values, color="#1170aa")
    ax.invert_yaxis()
    ax.set_xlabel("Detections")
    ax.set_title("Species detections")
    ax.grid(alpha=0.3, axis="x")
    for i, v in enumerate(values):
        ax.text(v + 0.1, i, str(v), va="center", fontsize=8)
    return _save(fig, out_path)


def coral_condition_donut(distribution: Mapping[str, float], out_path: str | Path) -> Path:
    """Donut of the 5 coral condition classes."""
    labels = [c for c in CLASS_ORDER if distribution.get(c, 0) > 0] or ["healthy"]
    sizes = [max(distribution.get(c, 0.0), 0.0) for c in labels] or [1.0]
    colors = [CLASS_COLORS[c] for c in labels]
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=[c.capitalize() for c in labels], colors=colors,
        autopct="%1.0f%%", startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor="white"),
        textprops=dict(fontsize=9),
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("black")
    ax.set_title("Coral condition distribution")
    return _save(fig, out_path)


def trend_chart(severity_history: Sequence[float], out_path: str | Path, ewma_alpha: float = 0.3) -> Path:
    """BSI trajectory with EWMA overlay (PDF-ready variant)."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    steps = np.arange(1, len(severity_history) + 1)
    ax.plot(steps, severity_history, "o-", ms=3, color="#1170aa", label="BSI")
    if len(steps) >= 2:
        ewma = np.array(severity_history, dtype=float)
        smoothed = [ewma[0]]
        for v in ewma[1:]:
            smoothed.append(ewma_alpha * v + (1 - ewma_alpha) * smoothed[-1])
        ax.plot(steps, smoothed, "--", color="#d55e00", lw=2, label="EWMA")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Bleaching severity index")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Bleaching severity trend")
    return _save(fig, out_path)


def score_card_charts(analytics_dict: Mapping, out_dir: str | Path) -> dict[str, Path]:
    """Score bars for MBS / RHS / env risk (compact, side-by-side in the PDF)."""
    out = ensure_dir(out_dir)
    scores = {
        "Biodiversity\n(MBS)": analytics_dict["biodiversity"]["mbs"],
        "Reef Health\n(RHS)": analytics_dict["reef_health"]["score"],
        "Env. Risk": analytics_dict["env_risk"]["score"],
        "Coral Risk": analytics_dict["coral_risk_score"],
    }
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    colors = ["#2ecc71" if v >= 70 else "#f1c40f" if v >= 40 else "#e74c3c" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (0-100)")
    ax.set_title("ReefScan composite scores")
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    return {"score_card": _save(fig, out / "score_card.png")}
