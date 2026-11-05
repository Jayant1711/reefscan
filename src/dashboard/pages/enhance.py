"""Enhancement comparison page: all strategies side-by-side with live metrics."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from ...enhancement.metrics import evaluate_enhancement
from ..pipeline_singleton import get_enhancers
from ..ui_components import hero, info_note


@st.cache_data(show_spinner="Benchmarking strategies ...")
def benchmark(_cfg_params: dict, img_bytes: bytes) -> tuple[list, list[dict]]:
    import io

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    enhancers = get_enhancers()
    rows = []
    panels = [("Original", image)]
    for enh in enhancers:
        try:
            out = enh.enhance(image)
        except Exception as exc:  # pragma: no cover
            st.warning(f"{enh.name} failed: {exc}")
            continue
        m = evaluate_enhancement(image, out)
        rows.append({"strategy": enh.name, **{k: round(v, 3) if np.isfinite(v) else None for k, v in m.items()}})
        panels.append((enh.name, out))
    return panels, rows


def page_enhance(cfg) -> None:
    hero("✨ Enhancement Comparison", "Phase 1 — 8 underwater enhancement strategies, scored live with PSNR · SSIM · UCIQE · UIQM")

    uploaded = st.file_uploader("Upload an underwater image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="enh")
    if uploaded is None:
        info_note("Upload an image to benchmark every enhancement strategy on it. "
                  "Metrics update live; the table ranks by the composite selection score.")
        return

    panels, rows = benchmark(dict(cfg.enhancement.params), uploaded.getvalue())
    if not rows:
        st.error("No strategy produced output.")
        return

    # --- side-by-side gallery
    cols = st.columns(3)
    for i, (name, img) in enumerate(panels):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cols[i % 3].image(rgb, caption=name, use_container_width=True)

    # --- metrics table
    import pandas as pd

    df = pd.DataFrame(rows)
    weights = cfg.enhancement.weights
    for metric in ("uciqe", "uiqm", "ssim"):
        span = df[metric].max() - df[metric].min()
        df[f"{metric}_norm"] = (df[metric] - df[metric].min()) / span if span > 1e-9 else 0.5
    df["score"] = (weights.uciqe * df.uciqe_norm + weights.uiqm * df.uiqm_norm + weights.ssim * df.ssim_norm).round(4)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    st.markdown("### Metrics (ranked by composite score)")
    st.dataframe(df.drop(columns=["uciqe_norm", "uiqm_norm", "ssim_norm"]), use_container_width=True)
    st.success(f"🏆 Best strategy for this image: **{df.iloc[0]['strategy']}** "
               f"(score {df.iloc[0]['score']:.4f})")
    st.caption("PSNR/SSIM measure restoration proximity to the raw capture; UCIQE/UIQM are no-reference "
               "underwater quality scores. Selection: 0.40·UCIQE + 0.30·UIQM + 0.30·SSIM (min-max normalised).")
