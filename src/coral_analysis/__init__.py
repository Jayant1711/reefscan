"""Phase 3 — Coral bleaching analysis: binary CNN + WGBS severity + risk + reports."""

from .severity import (
    SEVERITY_CLASSES,
    SEVERITY_WEIGHTS,
    BleachingAssessor,
    SeverityAssessment,
    index_to_class,
    severity_index,
    whiteness_ratio,
)
from .risk import coral_risk_score, risk_category
from .trend import BleachingTrendAnalyzer
from .health_report import build_health_report_markdown

__all__ = [
    "SEVERITY_CLASSES",
    "SEVERITY_WEIGHTS",
    "BleachingAssessor",
    "SeverityAssessment",
    "index_to_class",
    "severity_index",
    "whiteness_ratio",
    "coral_risk_score",
    "risk_category",
    "BleachingTrendAnalyzer",
    "build_health_report_markdown",
]
