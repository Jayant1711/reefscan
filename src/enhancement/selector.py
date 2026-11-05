"""Automatic enhancement strategy selection.

Runs every registered strategy on a sample of images, computes PSNR/SSIM/
UCIQE/UIQM, min-max normalises the no-reference scores across strategies, and
blends them into a composite score:

    score = w_uciqe*UCIQE_norm + w_uiqm*UIQM_norm + w_ssim*SSIM_norm

The winning strategy is persisted to ``results/enhancement/enhancement_report.json``
and reused by the inference pipeline and dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ..utils import ensure_dir, save_json, image_grid
from .metrics import evaluate_enhancement
from .registry import build_enhancers
from .base import Enhancer

logger = logging.getLogger(__name__)


class EnhancementSelector:
    """Benchmark and pick the best enhancement strategy for a collection."""

    def __init__(self, weights: dict[str, float] | None = None, strategies: Sequence[str] | None = None,
                 params: dict | None = None) -> None:
        self.weights = weights or {"uciqe": 0.40, "uiqm": 0.30, "ssim": 0.30}
        self.enhancers: list[Enhancer] = build_enhancers(list(strategies) if strategies else None, params)

    # ------------------------------------------------------------------ core
    def evaluate_images(self, images: Sequence[np.ndarray]) -> pd.DataFrame:
        """Evaluate every strategy on every image; return per-strategy means."""
        rows: list[dict] = []
        for enh in self.enhancers:
            per_image = []
            for img in images:
                try:
                    out = enh.enhance(img)
                    m = evaluate_enhancement(img, out)
                    per_image.append(m)
                except Exception as exc:  # keep benchmarking other strategies
                    logger.warning("Strategy '%s' failed on an image: %s", enh.name, exc)
            if not per_image:
                continue
            mean = {k: float(np.nanmean([m[k] if np.isfinite(m[k]) else np.nan for m in per_image]))
                    for k in per_image[0]}
            rows.append({"strategy": enh.name, **mean})
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No enhancement strategy produced results.")
        return df

    def _composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalise metric columns and compute the weighted score."""
        df = df.copy()
        for metric in ("uciqe", "uiqm", "ssim"):
            col = df[metric].to_numpy(dtype=np.float64)
            span = np.nanmax(col) - np.nanmin(col)
            df[f"{metric}_norm"] = (col - np.nanmin(col)) / span if span > 1e-9 else 0.5
        df["score"] = (
            self.weights["uciqe"] * df["uciqe_norm"]
            + self.weights["uiqm"] * df["uiqm_norm"]
            + self.weights["ssim"] * df["ssim_norm"]
        )
        return df.sort_values("score", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------- API
    def select_best(
        self,
        images: Sequence[np.ndarray],
        out_dir: str | Path | None = None,
        make_grid: bool = True,
    ) -> tuple[str, pd.DataFrame, Path | None]:
        """Return (best_strategy_name, ranked_report_df, comparison_grid_path)."""
        df = self._composite_scores(self.evaluate_images(images))
        best = str(df.iloc[0]["strategy"])
        logger.info("Best enhancement strategy: %s (score=%.4f)", best, df.iloc[0]["score"])

        grid_path: Path | None = None
        if out_dir is not None and make_grid:
            out = ensure_dir(out_dir)
            grid_path = self._comparison_grid(images, out)

        return best, df, grid_path

    def _comparison_grid(self, images: Sequence[np.ndarray], out_dir: Path, max_images: int = 2) -> Path:
        """One grid row per sample image: original + every strategy."""
        import cv2

        row_images: list[np.ndarray] = []
        titles: list[str] = []
        for img in list(images)[:max_images]:
            row_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            titles.append("Original")
            for enh in self.enhancers:
                try:
                    out = enh.enhance(img)
                except Exception:
                    out = img.copy()
                row_images.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                titles.append(enh.name)
        return image_grid(row_images, titles=titles, out_path=out_dir / "enhancement_comparison.png", cols=len(self.enhancers) + 1)

    def save_report(self, df: pd.DataFrame, best: str, out_dir: str | Path) -> Path:
        """Persist the ranked table (CSV) and selection summary (JSON)."""
        out = ensure_dir(out_dir)
        csv_path = out / "enhancement_benchmark.csv"
        df.to_csv(csv_path, index=False)

        top = df.iloc[0]
        save_json(
            {
                "best_strategy": best,
                "weights": self.weights,
                "metrics_best": {k: (None if not np.isfinite(top[k]) else float(top[k]))
                                 for k in ("psnr", "ssim", "uciqe", "uiqm", "score")},
                "ranking": df[["strategy", "score"]].to_dict(orient="records"),
            },
            out / "enhancement_report.json",
        )
        return csv_path


def run_enhancement_benchmark(
    images: Iterable[np.ndarray],
    out_dir: str | Path,
    weights: dict[str, float] | None = None,
    strategies: Sequence[str] | None = None,
    params: dict | None = None,
) -> tuple[str, pd.DataFrame]:
    """Convenience: benchmark strategies on ``images``, save report, return best."""
    selector = EnhancementSelector(weights=weights, strategies=strategies, params=params)
    best, df, _grid = selector.select_best(list(images), out_dir=out_dir)
    selector.save_report(df, best, out_dir)
    return best, df
