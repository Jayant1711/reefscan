"""Bleaching trend analytics helper operating on FrameAnalysis sequences."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..coral_analysis.trend import BleachingTrendAnalyzer, TrendSummary
from ..realtime.schema import FrameAnalysis


def severity_trend_summary(analyses: Sequence[FrameAnalysis], ewma_alpha: float = 0.3) -> TrendSummary:
    """Roll up per-frame BSI values into a trend summary (Phase 6 #4)."""
    trend = BleachingTrendAnalyzer(ewma_alpha)
    for analysis in analyses:
        if analysis.coral is not None:
            trend.update(analysis.coral.assessment.index)
    return trend.summary()
