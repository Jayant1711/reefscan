"""Reports page: generate and download PDF reports from dashboard analyses."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from ..ui_components import hero, info_note


def page_reports(cfg) -> None:
    hero("📄 Reports", "Phase 8 — publication-style PDF reports with images, detections, severity gauges and recommendations")

    info_note(
        "Reports are generated from the analyses you run in this dashboard. "
        "Analyse an image (Species/Coral page) or a survey set (Analytics page) first, then generate."
    )

    analytics = st.session_state.get("last_analytics")

    if st.button("🧾 Generate survey PDF report", type="primary",
                 disabled=not analytics, use_container_width=True):
        if not analytics:
            st.warning("Run the Analytics page first.")
            return
        with st.spinner("Composing PDF ..."):
            from ...analytics.survey import SurveyAnalytics  # noqa: F401 (structure reference)
            from ...reporting.pdf_report import ReefScanReportBuilder

            out_dir = Path(tempfile.mkdtemp(prefix="reefscan_report_"))
            # Rehydrate a light SurveyAnalytics-shaped dict through the builder's dict path.
            builder = ReefScanReportBuilder(cfg)
            out_path = out_dir / f"ReefScan_Survey_Report_{datetime.now():%Y%m%d_%H%M}.pdf"
            _render_survey_pdf(builder, analytics, out_path)
        st.success("Report generated.")
        st.download_button("⬇️ Download PDF", out_path.read_bytes(), out_path.name,
                           mime="application/pdf", use_container_width=True)
        st.caption(f"Also saved CLI-side via `python inference.py` / `evaluate.py` → reports/ folder.")


def _render_survey_pdf(builder, data: dict, out_path: Path) -> None:
    """Render the survey report from the analytics dict stored in session state."""
    import numpy as np

    from ...reporting.charts import coral_condition_donut, species_distribution_chart, trend_chart, score_card_charts

    assets = out_path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    counts = {r["species"]: r["count"] for r in data["population"]["table"]}
    species_chart = species_distribution_chart(counts, assets / "species.png")
    donut = coral_condition_donut(data.get("condition_distribution",
                                           {"healthy": 1.0}), assets / "donut.png")
    trend_png = trend_chart(data["trend"].get("history", [data["trend"]["last_index"]]), assets / "trend.png")
    score_card = score_card_charts(data, assets)["score_card"]

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib import colors

    story = []
    builder._cover(story, "Dashboard Survey Report", {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Reef Health Score": f"{data['reef_health']['score']:.0f}/100 ({data['reef_health']['category']})",
        "Biodiversity (MBS)": f"{data['biodiversity']['mbs']:.0f}/100",
    })
    builder._section(story, "1. Composite scores")
    builder._image_row(story, [score_card], ["MBS / RHS / risk scores"], width_each=14 * cm)
    builder._section(story, "2. Species population")
    builder._image_row(story, [species_chart], ["Detections per species"], width_each=10 * cm)
    if counts:
        rows = [["Species", "Count", "Share %"]] + [
            [r["species"], str(r["count"]), f"{r['relative_abundance_pct']:.1f}"]
            for r in data["population"]["table"]
        ]
        t = Table(rows, colWidths=[8 * cm, 3 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5FA5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9DCEC")),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
    builder._section(story, "3. Coral bleaching")
    builder._image_row(story, [donut, trend_png], ["Condition distribution", "Severity trend"], width_each=7.2 * cm)
    story.append(builder._kv_table([
        ("Trend", f"{data['trend']['trend_label']} (slope {data['trend']['slope_per_step']:+.2f}/obs)"),
        ("Environmental risk", f"{data['env_risk']['category']} ({data['env_risk']['score']:.0f}/100)"),
        ("Drivers", " · ".join(data["env_risk"]["drivers"])),
    ]))
    builder._section(story, "4. Data notes")
    story.append(Paragraph(
        "Generated from dashboard analyses (demo models may have been used — see warning banners). "
        "Formulas: docs/RESEARCH_METHODOLOGY.md.", builder.styles["SmallGrey"]))

    doc = SimpleDocTemplate(str(out_path), pagesize=A4, topMargin=2 * cm, bottomMargin=1.8 * cm)
    doc.build(story, onFirstPage=builder._header_footer, onLaterPages=builder._header_footer)
