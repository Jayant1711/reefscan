"""PDF report generator (reportlab platypus) — Phase 8.

Two report types:
  1. Single-image report  : original/enhanced/annotated/heatmap + detections + coral verdict
  2. Survey report        : analytics bundle (MBS/RHS/trends/population) + charts + actions
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..analytics.survey import SurveyAnalytics
from ..coral_analysis.severity import SEVERITY_CLASSES
from ..realtime.pipeline import FrameAnalysis
from ..utils import ensure_dir
from .charts import coral_condition_donut, score_card_charts, species_distribution_chart, trend_chart

logger = logging.getLogger(__name__)

BRAND = colors.HexColor("#0B5FA5")
LIGHT = colors.HexColor("#EAF3FB")
GREY = colors.HexColor("#5A6B7A")


class ReefScanReportBuilder:
    """Assembles ReefScan PDF reports with consistent branding."""

    def __init__(self, cfg: Any = None) -> None:
        self.cfg = cfg or {}
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle("BrandTitle", parent=self.styles["Title"], textColor=BRAND, fontSize=22, spaceAfter=6))
        self.styles.add(ParagraphStyle("H2Brand", parent=self.styles["Heading2"], textColor=BRAND, spaceBefore=14))
        self.styles.add(ParagraphStyle("SmallGrey", parent=self.styles["Normal"], textColor=GREY, fontSize=8))
        self.styles.add(ParagraphStyle("Center", alignment=TA_CENTER))

    # ------------------------------------------------------------------ core
    def _header_footer(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(BRAND)
        canvas.rect(0, A4[1] - 1.1 * cm, A4[0], 1.1 * cm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(1.5 * cm, A4[1] - 0.72 * cm, "ReefScan AI — Underwater Marine Assessment")
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawCentredString(A4[0] / 2, 0.9 * cm, f"Generated {datetime.now():%Y-%m-%d %H:%M} · Page {doc.page}")
        canvas.restoreState()

    def _cover(self, story: list, subtitle: str, meta: Mapping[str, str]) -> None:
        story.extend([
            Spacer(1, 5 * cm),
            Paragraph("ReefScan AI", self.styles["BrandTitle"]),
            Paragraph(subtitle, ParagraphStyle("sub", parent=self.styles["Normal"], fontSize=13, alignment=TA_CENTER, textColor=GREY)),
            Spacer(1, 2.2 * cm),
        ])
        rows = [[Paragraph(f"<b>{k}</b>", self.styles["Normal"]), Paragraph(v, self.styles["Normal"])] for k, v in meta.items()]
        table = Table(rows, colWidths=[5 * cm, 11 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9DCEC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(KeepTogether([table]))
        story.append(PageBreak())

    def _section(self, story: list, title: str) -> None:
        story.append(Paragraph(title, self.styles["H2Brand"]))

    def _image_row(self, story: list, paths: Sequence[Path | None], captions: Sequence[str], width_each: float = 4.6 * cm) -> None:
        cells = []
        for p, cap in zip(paths, captions):
            if p is not None and Path(p).exists():
                img = RLImage(str(p), width=width_each, height=width_each * 0.75)
                cells.append([img, Paragraph(cap, self.styles["SmallGrey"])])
            else:
                cells.append([Paragraph(f"<i>({cap}: unavailable)</i>", self.styles["SmallGrey"]), ""])
        data = [[c[0] for c in cells], [c[1] for c in cells]]
        table = Table(data, colWidths=[width_each + 0.3 * cm] * len(cells))
        table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(KeepTogether([table, Spacer(1, 0.3 * cm)]))

    def _kv_table(self, rows: Sequence[Sequence[str]]) -> Table:
        table = Table([[Paragraph(f"<b>{a}</b>"), Paragraph(str(b))] for a, b in rows], colWidths=[6.2 * cm, 9.8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9DCEC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    # --------------------------------------------------------------- reports
    def build_image_report(
        self,
        image_path: str | Path,
        analysis: FrameAnalysis,
        out_path: str | Path,
        assets: Mapping[str, Path | None] | None = None,
    ) -> Path:
        """Report for one analysed image.

        ``assets`` maps: original / enhanced / annotated / heatmap -> file paths.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assets = assets or {}
        coral = analysis.coral.assessment if analysis.coral else None

        story: list = []
        self._cover(story, "Single-Image Assessment Report", {
            "Source image": Path(image_path).name,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Analysed by": "ReefScan AI v1.0 (enhancement + CNN + WGBS)",
            "Enhancement strategy": getattr(analysis, "enhancer_name", "auto"),
        })

        self._section(story, "1. Visual results")
        self._image_row(story,
                        [assets.get("original"), assets.get("enhanced"), assets.get("annotated"), assets.get("heatmap")],
                        ["Original", "Enhanced", "Detections", "CAM heatmap"])

        self._section(story, "2. Species detections")
        if analysis.detections:
            rows = [["#", "Species", "Confidence", "Box (x1,y1,x2,y2)", "Track ID"]]
            for i, det in enumerate(analysis.detections, 1):
                rows.append([str(i), det.species, f"{det.confidence:.1%}",
                             f"({det.x1},{det.y1},{det.x2},{det.y2})",
                             str(det.track_id) if det.track_id is not None else "-"])
            t = Table(rows, colWidths=[1 * cm, 6 * cm, 2.8 * cm, 5 * cm, 2.2 * cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9DCEC")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ]))
            story.append(t)
        else:
            story.append(Paragraph("No fish were detected in this image.", self.styles["Normal"]))

        self._section(story, "3. Coral assessment (WGBS)")
        if coral is not None:
            gauge_path = out_path.parent / "assets" / "severity_gauge.png"
            from ..utils.visualization import severity_gauge as make_gauge

            make_gauge(coral.index, gauge_path)
            self._image_row(story, [gauge_path], [f"Severity index {coral.index:.1f}/100"], width_each=6.4 * cm)
            story.append(self._kv_table([
                ("Condition", coral.label),
                ("Severity index (BSI)", f"{coral.index:.1f} / 100"),
                ("P(bleached) from CNN", f"{coral.p_bleached:.2f}"),
                ("Whiteness ratio (physics)", f"{coral.whiteness:.2f}"),
                ("Recommendations", " · ".join(coral.recommendations) or "-"),
            ]))

        self._section(story, "4. Method notes")
        story.extend([
            Paragraph("Enhancement strategies were benchmarked with PSNR/SSIM/UCIQE/UIQM and the best "
                      "strategy selected automatically before inference (Phase 1).", self.styles["Normal"]),
            Paragraph("Bleaching severity fuses the CNN's bleached-probability with a whiteness ratio "
                      "measured in HSV space (Phase 3, WGBS). Heatmaps from GradCAM-class methods "
                      "explain the species prediction (Phase 4).", self.styles["Normal"]),
        ])

        doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=2 * cm, bottomMargin=1.8 * cm)
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        logger.info("PDF report -> %s", out_path)
        return out_path

    def build_survey_report(
        self,
        analytics: SurveyAnalytics,
        out_path: str | Path,
        site_name: str = "Survey",
        artifacts: Mapping[str, Path | None] | None = None,
    ) -> Path:
        """Report for a video/survey with the full Phase-6 analytics bundle."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        assets_dir = ensure_dir(out_path.parent / "assets")
        artifacts = artifacts or {}
        data = analytics.to_dict()

        # Pre-render chart assets.
        counts = {r["species"]: r["count"] for r in data["population"]["table"]}
        species_chart = species_distribution_chart(counts, assets_dir / "species_distribution.png")
        donut = coral_condition_donut(_dist_from_analytics(analytics), assets_dir / "coral_donut.png")
        trend_png = trend_chart(analytics.trend.history, assets_dir / "trend.png")
        score_card = score_card_charts(data, assets_dir)["score_card"]

        story: list = []
        self._cover(story, "Reef Survey & Health Report", {
            "Site": site_name,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Frames analysed": str(data["population"].get("frames", analytics.trend.n_points)),
            "Reef Health Score": f"{data['reef_health']['score']:.0f}/100 ({data['reef_health']['category']})",
        })

        self._section(story, "1. Composite scores")
        self._image_row(story, [score_card], ["MBS / RHS / Env risk / Coral risk"], width_each=14 * cm)
        story.append(self._kv_table([
            ("Marine Biodiversity Score", f"{data['biodiversity']['mbs']:.1f}/100 "
             f"(Shannon {data['biodiversity']['shannon']:.2f}, {data['biodiversity']['n_species']} species)"),
            ("Reef Health Score", f"{data['reef_health']['score']:.1f}/100 — {data['reef_health']['category']}"),
            ("Environmental risk", f"{data['env_risk']['category']} ({data['env_risk']['score']:.0f}/100)"),
            ("Bleaching risk", f"{data['coral_risk_category']} ({data['coral_risk_score']:.0f}/100)"),
            ("Trend", f"{data['trend']['trend_label']} (slope {data['trend']['slope_per_step']:+.2f}/obs)"),
        ]))

        self._section(story, "2. Species population")
        self._image_row(story, [species_chart], ["Detections per species"], width_each=10 * cm)
        if counts:
            rows = [["Species", "Count", "Share %"]]
            for r in data["population"]["table"]:
                rows.append([r["species"], str(r["count"]), f"{r['relative_abundance_pct']:.1f}"])
            t = Table(rows, colWidths=[8 * cm, 3 * cm, 3 * cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9DCEC")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ]))
            story.append(t)
        if data["population"]["rare_species"]:
            story.append(Paragraph(
                f"<b>Rare species ({len(data['population']['rare_species'])}):</b> "
                + ", ".join(data["population"]["rare_species"]), self.styles["Normal"]))

        self._section(story, "3. Coral bleaching")
        self._image_row(story, [donut, trend_png], ["Condition distribution", "Severity trend"], width_each=7.2 * cm)
        story.append(self._kv_table([
            ("Mean severity index", f"{analytics.trend.last_index:.1f} (last obs) / EWMA {analytics.trend.ewma_last:.1f}"),
            ("Frames with bleaching", f"{100 * analytics.bleached_frame_fraction:.0f}%"),
            ("Environment drivers", " · ".join(data["env_risk"]["drivers"])),
        ]))

        self._section(story, "4. Recommendations")
        for action in _survey_recommendations(analytics):
            story.append(Paragraph(f"•  {action}", self.styles["Normal"]))

        if artifacts.get("video_out"):
            self._section(story, "Appendix: annotated video")
            story.append(Paragraph(f"Annotated survey video: {Path(artifacts['video_out']).name}", self.styles["Normal"]))

        doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=2 * cm, bottomMargin=1.8 * cm)
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        logger.info("Survey PDF report -> %s", out_path)
        return out_path


def _dist_from_analytics(analytics: SurveyAnalytics) -> dict[str, float]:
    """Condition distribution: computed by SurveyAnalytics (config thresholds)."""
    dist = getattr(analytics, "condition_distribution", None)
    if dist:
        return dist
    # Legacy fallback: recompute from the trend history with default thresholds.
    from ..coral_analysis.severity import index_to_class_strict

    dist = {c: 0.0 for c in SEVERITY_CLASSES}
    for value in analytics.trend.history:
        cls = index_to_class_strict(value)
        dist[cls] += 1.0
    total = sum(dist.values()) or 1.0
    return {c: v / total for c, v in dist.items()}


def _survey_recommendations(analytics: SurveyAnalytics) -> list[str]:
    data = analytics.to_dict()
    recs: list[str] = []
    if data["reef_health"]["score"] >= 70:
        recs.append("Reef condition is favourable — continue scheduled monitoring.")
    else:
        recs.append("Reef condition is degraded — prioritise this site in the monitoring plan.")
    if data["trend"]["trend_label"] in {"Declining", "Critical"}:
        recs.append("Severity is rising across the survey — investigate thermal stress and local stressors.")
    if data["env_risk"]["category"] in {"High", "Critical"}:
        recs.append("Water conditions impair visibility — repeat survey under better conditions before final judgement.")
    if data["biodiversity"]["n_species"] <= 3:
        recs.append("Low species richness observed — extend survey area or duration for a robust biodiversity estimate.")
    if data["population"]["rare_species"]:
        recs.append("Rare species present — flag detections for expert verification.")
    return recs
