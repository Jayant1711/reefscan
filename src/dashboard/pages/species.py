"""Species classification page: upload image -> detections + species + confidence."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from ...realtime.annotate import annotate_frame
from ..pipeline_singleton import demo_warning_banner, get_pipeline
from ..ui_components import hero


def page_species(cfg) -> None:
    hero("🐟 Species Classification", "Phase 2/5 — upload an underwater image: enhancement → fish boxes → species + confidence")

    demo_warning_banner()

    col_upload, col_opts = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Underwater image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="sp")
    with col_opts:
        show_heatmap = st.checkbox("Overlay CAM heatmap", value=False, help="Slower — runs GradCAM on the top detection region.")

    if uploaded is None:
        st.info("Upload an underwater image to analyse it.")
        return

    arr = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    with st.spinner("Analysing ..."):
        pipeline = get_pipeline(enable_cam=False)
    if pipeline.fish_model is None:
        st.error("No trained fish model found — run `python run_smoke_test.py` (demo) or "
                 "`python train.py --task fish` first.", icon="🚨")
        return
    analysis = pipeline.analyze_image(image)
    heatmap = None
    if show_heatmap and analysis.detections:
        from ...explainability.cam import GradCAM
        from ...explainability.targets import find_target_layer

        with GradCAM(pipeline.fish_model, find_target_layer(pipeline.fish_model)) as cam:
            tensor = pipeline._preprocess(analysis.enhanced, pipeline.fish_img_size)
            heatmap = cam.generate(tensor)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    annotated = annotate_frame(image, analysis.detections,
                               analysis.coral.assessment if analysis.coral else None,
                               heatmap)

    left, right = st.columns([1.2, 1])
    with left:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated result", use_container_width=True)
        png = cv2.imencode(".png", annotated)[1].tobytes()
        st.download_button("⬇️ Download annotated image", png, "reefscan_annotated.png", mime="image/png",
                           use_container_width=True)
    with right:
        if analysis.detections:
            st.markdown("### Detections")
            for det in analysis.detections:
                st.markdown(
                    f"**{det.species}** — {det.confidence:.1%} · box ({det.x1},{det.y1},{det.x2},{det.y2})"
                    + (f" · track #{det.track_id}" if det.track_id is not None else "")
                )
            best = max(analysis.detections, key=lambda d: d.confidence)
            if analysis.species_probs:
                st.markdown("#### Top-5 probabilities (primary region)")
                for name, prob in analysis.species_probs.items():
                    st.progress(min(prob, 1.0), text=f"{name}: {prob:.1%}")
        else:
            st.info("No fish detected.")
        if analysis.coral is not None:
            st.markdown("### Coral status")
            a = analysis.coral.assessment
            st.metric("Bleaching severity (BSI)", f"{a.index:.1f}/100", a.label)
            st.caption(f"P(bleached)={a.p_bleached:.2f} · whiteness={a.whiteness:.2f}")
        if analysis.timings:
            st.caption(" · ".join(f"{k} {v*1000:.0f} ms" for k, v in analysis.timings.items()))
