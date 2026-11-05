"""Real-time page: video upload → frame-by-frame analysis → annotated MP4 + stats."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ..pipeline_singleton import demo_warning_banner, get_pipeline
from ..ui_components import hero, info_note


def page_realtime(cfg) -> None:
    hero("🎬 Upload Video", "Phase 5 — frame-by-frame analysis: species detections, unique fish counting, coral trend")

    demo_warning_banner()

    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if uploaded is None:
        info_note(
            "The pipeline analyses every Nth frame (configurable below), draws species boxes with unique "
            "track IDs, plots the coral severity banner, and writes an annotated MP4 + per-frame CSV."
        )
        return

    frame_skip = st.slider("Analyze every Nth frame (CPU speed)", 1, 10, int(cfg.realtime.frame_skip))
    max_frames = st.number_input("Max frames to analyze (0 = all)", 0, 100000, 300)

    if st.button("🚀 Run analysis", type="primary", use_container_width=True):
        tmp_dir = Path(tempfile.mkdtemp(prefix="reefscan_"))
        src = tmp_dir / uploaded.name
        src.write_bytes(uploaded.getvalue())
        out_path = tmp_dir / "annotated.mp4"

        pipeline = get_pipeline(enable_cam=False)
        pipeline.cfg.realtime.frame_skip = frame_skip
        pipeline.cfg.realtime.max_frames = int(max_frames)

        progress = st.progress(0.0, text="Analysing frames ...")
        last = {"frame": 0}

        def cb(frame_idx: int, total: int) -> None:
            last["frame"] = frame_idx
            if total:
                progress.progress(min(frame_idx / total, 1.0), text=f"Analysing frames ... {frame_idx}/{total}")

        with st.spinner("Processing video ..."):
            summary = pipeline.process_video(str(src), out_path, progress_cb=cb)
        progress.progress(1.0, text="Done.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames analysed", summary["frames_analyzed"])
        m2.metric("Unique fish", summary["unique_fish"])
        m3.metric("Species seen", len(summary["species_counts"]))
        m4.metric("Processing FPS", f"{summary['processing_fps']:.1f}")

        if summary.get("trend", {}).get("history"):
            st.markdown("### Bleaching severity across the video")
            st.line_chart(pd.DataFrame({
                "BSI": summary["trend"]["history"],
            }))
            st.caption(f"Trend: **{summary['trend']['trend_label']}** "
                       f"(slope {summary['trend']['slope_per_step']:+.2f}/obs)")

        if summary.get("species_counts"):
            st.markdown("### Species counts (unique tracks)")
            st.bar_chart(pd.Series(summary["species_counts"], name="detections"))

        if Path(out_path).exists():
            st.video(str(out_path))
            st.download_button("⬇️ Download annotated video", out_path.read_bytes(), "reefscan_annotated.mp4",
                               mime="video/mp4", use_container_width=True)
        if summary.get("frame_log") and Path(summary["frame_log"]).exists():
            csv_bytes = Path(summary["frame_log"]).read_bytes()
            st.download_button("⬇️ Download per-frame CSV", csv_bytes, "reefscan_frame_log.csv",
                               mime="text/csv", use_container_width=True)

        with st.expander("Raw summary JSON"):
            st.json(summary)

    st.divider()
    st.markdown("**Webcam:** run from a terminal with `python inference.py --mode webcam` "
                "(browser sandboxing blocks direct webcam capture here).")
