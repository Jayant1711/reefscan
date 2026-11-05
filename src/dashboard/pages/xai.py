"""XAI page: GradCAM / GradCAM++ / ScoreCAM heatmaps with downloads."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st
import torch

from ...explainability.cam import GradCAM, GradCAMPlusPlus, ScoreCAM
from ...explainability.targets import find_target_layer
from ...explainability.visualize import overlay_cam_on_image, save_heatmap
from ..pipeline_singleton import demo_warning_banner, get_pipeline
from ..ui_components import hero

METHODS = {
    "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus,
    "ScoreCAM": ScoreCAM,
}


def page_xai(cfg) -> None:
    hero("🔥 Heatmaps (Explainable AI)", "Phase 4 — GradCAM · GradCAM++ · ScoreCAM explain the species prediction")

    demo_warning_banner()

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="xai")
    if uploaded is None:
        st.info("Upload an underwater image; heatmaps show the image regions driving the species prediction.")
        return

    chosen = st.multiselect("Methods", list(METHODS), default=["GradCAM", "GradCAM++"])
    if cfg.xai.scorecam_max_channels:
        st.caption(f"ScoreCAM channel cap (CPU speed): {cfg.xai.scorecam_max_channels}")

    arr = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    pipeline = get_pipeline(enable_cam=False)
    analysis = pipeline.analyze_image(image)
    if not analysis.detections:
        st.warning("No fish region found — explaining the whole-frame prediction instead.")

    tensor = pipeline._preprocess(analysis.enhanced, pipeline.fish_img_size)
    with torch.no_grad():
        class_idx = int(pipeline.fish_model(tensor).argmax(dim=1).item())
    label = pipeline.fish_classes[class_idx]
    st.markdown(f"Explaining prediction: **{label}**")

    target_layer = find_target_layer(pipeline.fish_model)
    cols = st.columns(max(2, len(chosen)))
    for col, name in zip(cols, chosen):
        with col:
            with st.spinner(f"{name} ..."):
                method = METHODS[name](pipeline.fish_model, target_layer) if name != "ScoreCAM" else \
                    ScoreCAM(pipeline.fish_model, target_layer,
                             batch_size=cfg.xai.scorecam_batch,
                             max_channels=cfg.xai.scorecam_max_channels)
                with method:
                    cam = method.generate(tensor, class_idx)
            overlay = overlay_cam_on_image(analysis.enhanced, cam, cfg.xai.overlay_alpha)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=name, use_container_width=True)
            path = save_heatmap(cam, f"results/xai/{name.lower()}_heatmap.png")
            st.download_button(f"⬇️ {name} heatmap", path.read_bytes(), path.name,
                               mime="image/png", key=f"dl_{name}", use_container_width=True)
