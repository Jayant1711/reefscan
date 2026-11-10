"""Analytics page: multi-image survey -> MBS / RHS / trends / population / env risk."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from ...analytics.survey import SurveyAnalyzer
from ..pipeline_singleton import demo_warning_banner, get_pipeline
from ..ui_components import hero, info_note


def page_analytics(cfg) -> None:
    hero("📊 Analytics", "Phase 6 — Marine Biodiversity Score · Reef Health Score · population statistics · trends · environmental risk")

    demo_warning_banner()

    uploaded = st.file_uploader(
        "Upload survey images (multiple)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="analytics",
    )
    if not uploaded:
        info_note(
            "Upload several images from the same survey site. ReefScan aggregates the per-frame analyses "
            "into the Phase-6 research metrics: <b>MBS</b> (Shannon + Gini-Simpson + Margalef), "
            "<b>RHS</b> (severity + coverage + environment), population stats, bleaching trend and risk."
        )
        return
    if len(uploaded) > 40:
        st.warning("First 40 images will be analysed (CPU-friendly limit).")
        uploaded = uploaded[:40]

    pipeline = get_pipeline(enable_cam=False)
    if pipeline.fish_model is None:
        st.error("No trained fish model found — run `python run_smoke_test.py` (demo) or "
                     "`python train.py --task fish` first.", icon="🚨")
        return
    analyses, source_images = [], []
    progress = st.progress(0.0, text="Analysing images ...")
    for i, up in enumerate(uploaded):
        arr = np.frombuffer(up.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        source_images.append(img)
        analyses.append(pipeline.analyze_image(img, frame_idx=i, track=True))
        progress.progress((i + 1) / len(uploaded), text=f"Analysing images ... {i + 1}/{len(uploaded)}")
    progress.progress(1.0, text="Aggregating analytics ...")

    analyzer = SurveyAnalyzer(cfg)
    bundle = analyzer.analyze(analyses, source_images)
    data = bundle.to_dict()
    st.session_state["last_analytics"] = data  # shared with the Reports page

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Marine Biodiversity Score", f"{data['biodiversity']['mbs']:.0f}/100",
              f"{data['biodiversity']['n_species']} species")
    m2.metric("Reef Health Score", f"{data['reef_health']['score']:.0f}/100", data["reef_health"]["category"])
    m3.metric("Environmental risk", data["env_risk"]["category"], f"{data['env_risk']['score']:.0f}/100")
    m4.metric("Bleaching trend", data["trend"]["trend_label"], f"slope {data['trend']['slope_per_step']:+.1f}")

    left, right = st.columns(2)
    with left:
        st.markdown("### Species population")
        if data["population"]["table"]:
            st.bar_chart({r["species"]: r["count"] for r in data["population"]["table"]})
            st.dataframe(
                [{"Species": r["species"], "Count": r["count"], "Share %": r["relative_abundance_pct"]}
                 for r in data["population"]["table"]],
                use_container_width=True,
            )
            if data["population"]["rare_species"]:
                st.caption(f"Rare (<{cfg.analytics.rare_species_pct}% of detections): "
                           + ", ".join(data["population"]["rare_species"]))
        else:
            st.info("No fish detected in this set.")
    with right:
        st.markdown("### Coral bleaching")
        if data["trend"]["n_points"]:
            st.line_chart({"BSI": data["trend"].get("history", []) or [data["trend"]["last_index"]]})
        st.caption(f"Frames with visible bleaching: {100 * data['bleached_frame_fraction']:.0f}% · "
                   f"EWMA {data['trend']['ewma_last']:.1f}")
        st.markdown("**Environment drivers**")
        for driver in data["env_risk"]["drivers"]:
            st.markdown(f"- {driver}")

    st.divider()
    st.markdown("### Composite score definitions")
    st.caption(
        "MBS = 100·(0.40·H'/ln S_ref + 0.30·(1−D) + 0.30·min(d/d_ref,1)) · "
        "RHS = 100 − (0.60·BSI + 0.25·bleached% + 0.15·env-risk) · "
        "Env-risk = 0.45·turbidity + 0.25·(1−visibility) + 0.30·tanh(trend slope). "
        "Full derivations: docs/RESEARCH_METHODOLOGY.md"
    )
