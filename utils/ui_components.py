from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


def load_css(filepath: str | Path) -> None:
    path = Path(filepath)
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "AI RESEARCH WORKSPACE") -> None:
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-eyebrow">{escape(eyebrow)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(subtitle)}</p>
            <div class="hero-badges">
                <span>Sentence Transformers</span>
                <span>FAISS</span>
                <span>FLAN-T5</span>
                <span>NLLB-200</span>
                <span>Streamlit</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h2>{escape(title)}</h2>
            <p>{escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, text: str, icon: str = "✦") -> None:
    st.markdown(
        f"""
        <article class="glass-card">
            <div class="card-icon">{escape(icon)}</div>
            <h3>{escape(title)}</h3>
            <p>{escape(text)}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


def result_panel(title: str, content: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <section class="result-panel">
            <div class="result-label">{escape(title)}</div>
            <div class="result-content">{escape(content).replace(chr(10), "<br>")}</div>
            <div class="result-caption">{escape(caption)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def status_pill(label: str, state: str = "online") -> None:
    st.markdown(
        f'<div class="status-pill {escape(state)}"><span></span>{escape(label)}</div>',
        unsafe_allow_html=True,
    )
