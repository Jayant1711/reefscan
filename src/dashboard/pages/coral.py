"""Coral analysis page: bleaching severity, WGBS breakdown, recommendations."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from ...coral_analysis.risk import coral_risk_score
from ...utils.visualization import severity_gauge
from ..pipeline_singleton import demo_warning_banner, get_pipeline
from ..ui_components import hero, info_note, severity_chip


def page_coral(cfg) -> None:
    hero("🪸 Coral Analysis", "Phase 3 — binary coral CNN + Whiteness-Guided Bleaching Severity (WGBS) → 0–100 index, 5-class condition")

    demo_warning_banner()

    uploaded = st.file_uploader("Upload a coral patch image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="coral")
    if uploaded is None:
        info_note(
            "The CNN estimates P(bleached); the WGBS estimator measures the fraction of pale-bright "
            "pixels (low HSV saturation, high value). They fuse into "
            "<b>BSI = λ·100·P(bleached) + (1−λ)·100·whiteness</b>, mapped to Healthy/Mild/Moderate/Severe/Dead."
        )
        return

    arr = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    with st.spinner("Assessing coral condition ..."):
        pipeline = get_pipeline(enable_cam=False)
        analysis = pipeline.analyze_image(image)

    if analysis.coral is None:
        st.error("Coral assessment failed.")
        return
    a = analysis.coral.assessment
    risk = coral_risk_score(a.index)

    left, right = st.columns([1, 1.2])
    with left:
        st.image(cv2.cvtColor(analysis.enhanced, cv2.COLOR_BGR2RGB),
                 caption=f"Enhanced (strategy: {pipeline.enhancer_name})", use_container_width=True)
        severity_chip(a.class_name)
        st.metric("Bleaching severity index", f"{a.index:.1f} / 100", a.label)
    with right:
        import tempfile
        from pathlib import Path

        tmp = Path(tempfile.gettempdir()) / "reefscan_gauge.png"
        severity_gauge(a.index, tmp)
        st.image(str(tmp), use_container_width=True)

        st.markdown("### WGBS breakdown")
        st.progress(a.p_bleached, text=f"CNN P(bleached): {a.p_bleached:.2f} (λ = {cfg.coral.severity.fusion_lambda})")
        st.progress(a.whiteness, text=f"Whiteness ratio: {a.whiteness:.2f}")
        st.caption(f"Scene check — P(CORAL class)={analysis.coral.p_coral:.2f} "
                   f"({'coral scene' if analysis.coral.is_coral_scene else 'uncertain coral scene'})")

        st.markdown("### Risk")
        st.metric("Bleaching risk", f"{risk.score:.0f}/100", risk.category)
        st.markdown("**Recommendations**")
        for rec in a.recommendations:
            st.markdown(f"- {rec}")
