"""ReefScan AI — Streamlit dashboard entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when launched from the project root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import load_config
from src.utils.logger import setup_logging

st.set_page_config(
    page_title="ReefScan AI — Underwater Marine Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_logging()


@st.cache_resource(show_spinner="Loading configuration ...")
def cached_config():
    return load_config()


CFG = cached_config()

# Inject theme
theme_css = Path(__file__).parent / "src" / "dashboard" / "theme.css"
if theme_css.exists():
    st.markdown(f"<style>{theme_css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

from src.dashboard.pages import (  # noqa: E402  (after set_page_config)
    page_analytics,
    page_coral,
    page_enhance,
    page_home,
    page_realtime,
    page_reports,
    page_species,
    page_xai,
)

PAGES = {
    "🏠 Home": page_home,
    "🖼️ Upload Image": page_species,
    "🎬 Upload Video": page_realtime,
    "✨ Enhancement Comparison": page_enhance,
    "🐟 Species Classification": page_species,
    "🪸 Coral Analysis": page_coral,
    "🔥 Heatmaps (XAI)": page_xai,
    "📊 Analytics": page_analytics,
    "📄 Reports": page_reports,
}

with st.sidebar:
    st.markdown("## 🌊 ReefScan AI")
    st.caption("Underwater Marine Species & Coral Reef Health Assessment")
    choice = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption(
        "100% free stack: PyTorch · OpenCV · Streamlit · reportlab\n\n"
        "Works on CPU, Colab and Kaggle."
    )

PAGES[choice](CFG)
