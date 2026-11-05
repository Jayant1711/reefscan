"""Survey-level aggregation: turns a sequence of frame analyses into the full
ReefScan analytics bundle (biodiversity + reef health + population + trend + env risk)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..coral_analysis.risk import coral_risk_score
from ..coral_analysis.trend import TrendSummary
from ..realtime.schema import FrameAnalysis
from ..utils import read_image_bgr
from .biodiversity import marine_biodiversity_score
from .env_risk import EnvRiskResult, TurbidityMetrics, environmental_risk, measure_turbidity
from .population import species_population_stats
from .reef_health import ReefHealthResult, reef_health_score
from .trends import severity_trend_summary


@dataclass
class SurveyAnalytics:
    """Complete Phase-6 analytics bundle for one survey (images or a video)."""

    biodiversity: dict
    reef_health: ReefHealthResult
    population: dict
    trend: TrendSummary
    env_risk: EnvRiskResult
    coral_risk_score: float
    coral_risk_category: str
    bleached_frame_fraction: float

    def to_dict(self) -> dict:
        return {
            "biodiversity": self.biodiversity,
            "reef_health": {
                "score": self.reef_health.score,
                "category": self.reef_health.category,
                "components": self.reef_health.components,
            },
            "population": {
                "total_detections": self.population["total_detections"],
                "n_species": self.population["n_species"],
                "dominant_species": self.population["dominant_species"],
                "rare_species": self.population["rare_species"],
                "mean_detections_per_frame": self.population["mean_detections_per_frame"],
                "table": self.population["table"].to_dict(orient="records"),
            },
            "trend": {
                "n_points": self.trend.n_points,
                "first_index": self.trend.first_index,
                "last_index": self.trend.last_index,
                "ewma_last": self.trend.ewma_last,
                "slope_per_step": self.trend.slope_per_step,
                "trend_label": self.trend.trend_label,
            },
            "env_risk": {
                "score": self.env_risk.score,
                "category": self.env_risk.category,
                "drivers": self.env_risk.drivers,
                "turbidity_proxy": self.env_risk.turbidity.turbidity_proxy,
                "visibility_proxy": self.env_risk.turbidity.visibility_proxy,
            },
            "coral_risk_score": self.coral_risk_score,
            "coral_risk_category": self.coral_risk_category,
            "bleached_frame_fraction": self.bleached_frame_fraction,
        }


class SurveyAnalyzer:
    """Aggregates FrameAnalysis objects (images or video frames) into analytics."""

    def __init__(self, cfg: any = None) -> None:  # noqa: UP037 - cfg is a DotDict
        import math

        self.cfg = cfg
        a = cfg.analytics if cfg is not None else {}
        self.bio_weights = dict(getattr(a, "biodiversity_weights", {"shannon": 0.4, "simpson": 0.3, "margalef": 0.3}))
        self.shannon_ref = int(getattr(a, "shannon_ref_classes", 25))
        self.margalef_ref = float(getattr(a, "margalef_ref", 3.0))
        self.rh_weights = dict(getattr(a, "reef_health_weights", {"severity": 0.6, "bleached_frac": 0.25, "risk": 0.15}))
        self.rare_pct = float(getattr(a, "rare_species_pct", 1.0))

    def analyze(self, analyses: Sequence[FrameAnalysis], source_images: Sequence[np.ndarray] | None = None) -> SurveyAnalytics:
        if not analyses:
            raise ValueError("SurveyAnalyzer needs at least one FrameAnalysis.")

        # ---- population & biodiversity
        population = species_population_stats(analyses, rare_pct=self.rare_pct)
        counts = {row["species"]: int(row["count"]) for row in population["table"].to_dict(orient="records")}
        biodiversity = marine_biodiversity_score(
            counts, weights=self.bio_weights, shannon_ref_classes=self.shannon_ref, margalef_ref=self.margalef_ref
        )

        # ---- coral severity & trend
        bsi_values = [a.coral.assessment.index for a in analyses if a.coral is not None]
        mean_bsi = float(np.mean(bsi_values)) if bsi_values else 0.0
        bleached_frames = [a for a in analyses if a.coral is not None and a.coral.assessment.index >= 15]
        bleached_frac = len(bleached_frames) / max(1, len(bsi_values))
        trend = severity_trend_summary(analyses, getattr(self.cfg.analytics, "trend_ewma_alpha", 0.3) if self.cfg else 0.3)

        # ---- environment
        images = list(source_images) if source_images is not None else []
        if not images:
            images = [a.enhanced for a in analyses[:5]]
        turb_metrics = [measure_turbidity(img) for img in images if img is not None]
        avg_turb = TurbidityMetrics(
            turbidity_proxy=float(np.mean([t.turbidity_proxy for t in turb_metrics] or [0.0])),
            visibility_proxy=float(np.mean([t.visibility_proxy for t in turb_metrics] or [0.0])),
            mean_dark_channel=float(np.mean([t.mean_dark_channel for t in turb_metrics] or [0.0])),
            rms_contrast=float(np.mean([t.rms_contrast for t in turb_metrics] or [0.0])),
        )
        env = environmental_risk(avg_turb, trend.slope_per_step)

        # ---- scores
        coral_risk = coral_risk_score(mean_bsi, bleached_frac, avg_turb.turbidity_proxy, trend.slope_per_step)
        health = reef_health_score(mean_bsi, bleached_frac, env.score, weights=self.rh_weights)

        return SurveyAnalytics(
            biodiversity=biodiversity,
            reef_health=health,
            population=population,
            trend=trend,
            env_risk=env,
            coral_risk_score=coral_risk.score,
            coral_risk_category=coral_risk.category,
            bleached_frame_fraction=bleached_frac,
        )
