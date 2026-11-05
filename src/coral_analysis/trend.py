"""Bleaching trend analytics: EWMA smoothing + linear slope over a survey series."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import ensure_dir
from ..utils.visualization import _save  # matplotlib Agg backend guaranteed


@dataclass
class TrendSummary:
    n_points: int
    first_index: float
    last_index: float
    ewma_last: float
    slope_per_step: float
    trend_label: str            # Improving / Stable / Declining / Critical
    history: list[float] = field(default_factory=list)


class BleachingTrendAnalyzer:
    """Accumulates severity-index observations and summarises the trajectory."""

    def __init__(self, ewma_alpha: float = 0.3) -> None:
        self.ewma_alpha = ewma_alpha
        self.series: list[float] = []

    def update(self, severity_index: float) -> None:
        self.series.append(float(np.clip(severity_index, 0, 100)))

    def summary(self) -> TrendSummary:
        if not self.series:
            return TrendSummary(0, 0.0, 0.0, 0.0, 0.0, "Stable")
        arr = np.asarray(self.series, dtype=np.float64)
        ewma = arr[0]
        for x in arr[1:]:
            ewma = self.ewma_alpha * x + (1 - self.ewma_alpha) * ewma
        if len(arr) >= 2:
            slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
        else:
            slope = 0.0
        return TrendSummary(
            n_points=len(arr),
            first_index=float(arr[0]),
            last_index=float(arr[-1]),
            ewma_last=float(ewma),
            slope_per_step=slope,
            trend_label=self._label(arr[-1], slope),
            history=self.series.copy(),
        )

    @staticmethod
    def _label(last: float, slope: float) -> str:
        if last >= 60 and slope >= 0:
            return "Critical"
        if slope > 2:
            return "Declining"
        if slope < -2:
            return "Improving"
        return "Stable"

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"step": range(1, len(self.series) + 1), "severity_index": self.series})

    def plot(self, out_path: str | Path) -> Path:
        """Render the severity trajectory with an EWMA overlay."""
        import matplotlib.pyplot as plt

        df = self.to_dataframe()
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(df["step"], df["severity_index"], "o-", ms=4, label="BSI per observation", color="#1f77b4")
        if len(df) >= 2:
            ewma = df["severity_index"].ewm(alpha=self.ewma_alpha).mean()
            ax.plot(df["step"], ewma, "--", lw=2, label="EWMA", color="#d62728")
        for y, lab in ((15, "Mild"), (35, "Moderate"), (60, "Severe"), (85, "Dead")):
            ax.axhline(y, ls=":", lw=0.8, color="grey")
            ax.text(0.99, y + 1, lab, ha="right", fontsize=7, color="grey", transform=ax.get_yaxis_transform())
        ax.set_xlabel("Observation #")
        ax.set_ylabel("Bleaching severity index")
        ax.set_title("Coral bleaching trend")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        return _save(fig, ensure_dir(Path(out_path).parent) / Path(out_path).name)
