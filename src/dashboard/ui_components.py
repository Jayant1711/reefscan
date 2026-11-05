"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='reef-hero'><h1>{title}</h1><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def severity_chip(class_name: str) -> None:
    st.markdown(
        f"<span class='reef-chip chip-{class_name}'>{class_name.capitalize()}</span>",
        unsafe_allow_html=True,
    )


def info_note(text: str) -> None:
    st.markdown(f"<div class='reef-note'>{text}</div>", unsafe_allow_html=True)


def download_image_button(image_bytes: bytes, label: str, filename: str, key: str) -> None:
    st.download_button(label=label, data=image_bytes, file_name=filename,
                       mime="image/png", key=key, use_container_width=True)


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    """Row of st.metric cards; each entry is (label, value, delta)."""
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)
