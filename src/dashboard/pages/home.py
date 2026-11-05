"""Home page: system overview + live model status."""

from __future__ import annotations

import streamlit as st

from ..pipeline_singleton import model_status
from ..ui_components import hero, info_note, metric_row


def page_home(cfg) -> None:
    hero(
        "🌊 ReefScan AI",
        "Underwater Marine Species & Coral Reef Health Visual Assessment — "
        "enhancement · species classification · bleaching severity · explainability · analytics · reports",
    )

    status = model_status()
    fish_badge = {"trained": "✅ trained", "demo": "🧪 demo", "missing": "❌ missing"}.get(status.get("fish", "missing"))
    coral_badge = {"trained": "✅ trained", "demo": "🧪 demo", "missing": "❌ missing"}.get(status.get("coral", "missing"))
    seg_badge = "✅ trained" if status.get("seg") == "trained" else "➖ optional"
    enh_badge = status.get("enhancement", "default")

    metric_row([
        ("Fish classifier", fish_badge, None),
        ("Coral classifier", coral_badge, None),
        ("Fish segmenter", seg_badge, None),
        ("Enhancement", enh_badge, None),
    ])

    st.divider()
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("### How it works")
        st.markdown(
            """
1. **Enhance** — 8 classical strategies (CLAHE, Dark Channel Prior, UDCP, white balance,
   color restoration, gamma, histogram equalization, composite fusion) benchmarked with
   PSNR / SSIM / UCIQE / UIQM; the best is auto-selected.
2. **Detect & classify** — a lightweight U-Net (DeepFish) proposes fish regions; a
   transfer-learned CNN (FishNet taxonomy subset) labels each region.
3. **Assess coral** — a binary NOAA-trained CNN plus the Whiteness-Guided Bleaching
   Severity (WGBS) estimator produce a 0–100 severity index and 5-class condition.
4. **Explain** — GradCAM / GradCAM++ / ScoreCAM heatmaps show *where* the network looked.
5. **Aggregate** — Marine Biodiversity Score, Reef Health Score, population statistics,
   bleaching trends and environmental risk roll up into PDF reports.
"""
        )
    with right:
        st.markdown("### Quick start (CLI)")
        st.code(
            "python download_data.py            # fetch datasets\n"
            "python train.py --task fish --model resnet34\n"
            "python train.py --task coral --model resnet34\n"
            "python evaluate.py --task fish     # leaderboard\n"
            "python inference.py --mode image --input photo.jpg\n"
            "streamlit run app.py               # this dashboard",
            language="bash",
        )
        info_note(
            "No trained models yet? Run <b>python run_smoke_test.py</b> — it trains tiny "
            "demo checkpoints on synthetic data in minutes so every page of this dashboard works instantly."
        )

    st.divider()
    st.markdown("### Datasets")
    st.markdown(
        """
| Dataset | Role | Size |
|---|---|---|
| [FishNet (ICCV 2023)](https://fishnet-2023.github.io/) | Species classification (taxonomy-level subset) | 94,532 images · 17,357 species |
| [DeepFish (Sci. Reports 2020)](https://alzayats.github.io/DeepFish/) | Fish presence + segmentation (localization) | 39,766 frames · 620 masks |
| [NOAA PIFSC ESD Coral Bleaching](https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset) | Coral vs bleached coral (binary) → WGBS severity | 10,419 patches |
"""
    )
