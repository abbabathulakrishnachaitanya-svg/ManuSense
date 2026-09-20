"""
app/streamlit_app.py  —  ManuSense
====================================
AI-Powered Manufacturing Quality & Root-Cause Intelligence
"ManuSense doesn't just detect defects — it understands the manufacturing process behind them."

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as st_components
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ManuSense",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = float(settings.confidence_threshold)
if "session_counts" not in st.session_state:
    st.session_state.session_counts = {"total": 0, "ACCEPT": 0, "REJECT": 0, "REVIEW": 0}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
<div style="padding:4px 0 8px 0;">
  <div style="font-size:1.25rem;font-weight:900;letter-spacing:-0.3px;">
    🏭 Manu<span style="color:#e8500a;">Sense</span>
  </div>
  <div style="font-size:0.72rem;color:#64748b;margin-top:2px;">
    Manufacturing Quality Intelligence
  </div>
</div>
""", unsafe_allow_html=True)
    st.divider()

    # Dark mode toggle
    dark = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dm_toggle")
    st.session_state.dark_mode = dark

    st.divider()

    # Confidence threshold slider — no metric previews below it
    st.markdown("**⚙️ Decision Threshold**")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=st.session_state.conf_threshold,
        step=0.01,
        format="%.2f",
        help="Predictions at or above this value auto-ACCEPT (normal) or auto-REJECT (defect). Below → REVIEW.",
        key="threshold_slider",
        label_visibility="collapsed",
    )
    st.session_state.conf_threshold = threshold
    st.caption(f"Auto-decide at **{threshold:.0%}** confidence · below → Manual Review")

    st.divider()

    # Session summary — clean single-column layout
    sc = st.session_state.session_counts
    st.markdown("**📊 Session Summary**")

    st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:6px;margin:6px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:7px 10px;background:rgba(22,163,74,0.08);
              border-radius:7px;border-left:3px solid #16a34a;">
    <span style="font-size:0.82rem;font-weight:600;color:#166534;">✅ Accept</span>
    <span style="font-size:1.1rem;font-weight:800;color:#15803d;">{sc['ACCEPT']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:7px 10px;background:rgba(220,38,38,0.08);
              border-radius:7px;border-left:3px solid #dc2626;">
    <span style="font-size:0.82rem;font-weight:600;color:#991b1b;">❌ Reject</span>
    <span style="font-size:1.1rem;font-weight:800;color:#b91c1c;">{sc['REJECT']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:7px 10px;background:rgba(217,119,6,0.08);
              border-radius:7px;border-left:3px solid #d97706;">
    <span style="font-size:0.82rem;font-weight:600;color:#92400e;">🔍 Review</span>
    <span style="font-size:1.1rem;font-weight:800;color:#b45309;">{sc['REVIEW']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:7px 10px;background:rgba(100,116,139,0.08);
              border-radius:7px;border-left:3px solid #94a3b8;">
    <span style="font-size:0.82rem;font-weight:600;color:#475569;">Total</span>
    <span style="font-size:1.1rem;font-weight:800;color:#0f172a;">{sc['total']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if sc["total"] > 0:
        defect_rate = (sc["REJECT"] + sc["REVIEW"]) / sc["total"] * 100
        st.progress(min(int(defect_rate), 100))
        st.caption(f"Defect / Review rate: **{defect_rate:.0f}%**")

    if sc["total"] > 0:
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.history = []
            st.session_state.session_counts = {"total": 0, "ACCEPT": 0, "REJECT": 0, "REVIEW": 0}
            st.rerun()

    st.divider()
    st.caption("ManuSense v2.0 · ResNet18 · 99.11% accuracy")


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC THEME (dark / light)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.dark_mode:
    BG       = "#0f172a"
    BG2      = "#1e293b"
    BG3      = "#334155"
    TEXT     = "#f1f5f9"
    TEXT2    = "#94a3b8"
    BORDER   = "#334155"
    CARD_BG  = "#1e293b"
    METRIC_BG= "#1e293b"
else:
    BG       = "#ffffff"
    BG2      = "#f8fafc"
    BG3      = "#e2e8f0"
    TEXT     = "#0f172a"
    TEXT2    = "#64748b"
    BORDER   = "#e2e8f0"
    CARD_BG  = "#f8fafc"
    METRIC_BG= "#f8fafc"

# ─────────────────────────────────────────────────────────────────────────────
# CSS — KEYFRAMES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0);    }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-28px); }
    to   { opacity: 1; transform: translateX(0);     }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(28px); }
    to   { opacity: 1; transform: translateX(0);    }
}
@keyframes scaleIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1);    }
}
@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 0 0    rgba(22,163,74,0.55); }
    50%       { box-shadow: 0 0 0 14px rgba(22,163,74,0);    }
}
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 0 0    rgba(220,38,38,0.55); }
    50%       { box-shadow: 0 0 0 14px rgba(220,38,38,0);    }
}
@keyframes pulseAmber {
    0%, 100% { box-shadow: 0 0 0 0    rgba(217,119,6,0.55); }
    50%       { box-shadow: 0 0 0 14px rgba(217,119,6,0);    }
}
@keyframes shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position:  600px 0; }
}
@keyframes confFill { from { width: 0% !important; } }
@keyframes progressPop {
    from { opacity: 0; transform: scale(0.7); }
    to   { opacity: 1; transform: scale(1);   }
}
@keyframes countUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
@keyframes heroGlow {
    0%, 100% { box-shadow: 0 8px 40px rgba(232,80,10,0.18); }
    50%       { box-shadow: 0 8px 60px rgba(232,80,10,0.38); }
}
@keyframes scanBar {
    0%   { left: -40%; }
    100% { left: 100%; }
}
@keyframes rowIn {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
.anim-fadeInUp   { animation: fadeInUp   0.55s cubic-bezier(.22,.68,0,1.2) both; }
.anim-fadeIn     { animation: fadeIn     0.45s ease both; }
.anim-slideLeft  { animation: slideInLeft  0.5s cubic-bezier(.22,.68,0,1.2) both; }
.anim-slideRight { animation: slideInRight 0.5s cubic-bezier(.22,.68,0,1.2) both; }
.anim-scaleIn    { animation: scaleIn    0.45s cubic-bezier(.22,.68,0,1.2) both; }
.anim-d1 { animation-delay: 0.08s; }
.anim-d2 { animation-delay: 0.16s; }
.anim-d3 { animation-delay: 0.24s; }
.anim-d4 { animation-delay: 0.32s; }
.anim-d5 { animation-delay: 0.40s; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — DYNAMIC THEME VARS + MAIN STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
html, body, .stApp, .stApp > div,
section[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"] > div,
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"],
.block-container, .main, [data-testid="stHeader"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {BG2} !important;
    color: {TEXT} !important;
}}
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stVerticalBlock"] > div > p,
label {{ color: {TEXT} !important; }}

.ms-hero, .ms-hero *, .ms-hero p, .ms-hero span, .ms-hero div, .ms-hero h1 {{
    color: #ffffff !important;
}}
.ms-step {{ color: #ffffff !important; }}
.ms-step * {{ color: #ffffff !important; }}
.ms-step-num {{ color: #f97316 !important; font-weight: 800; }}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {{ color: #1a1a1a !important; }}

[data-testid="stMetric"] {{
    background-color: {METRIC_BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}}
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {{ color: {TEXT} !important; }}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"] div {{ color: {TEXT} !important; }}

.stTabs [data-baseweb="tab-list"] {{
    background-color: {BG2} !important;
    border-bottom: 2px solid {BORDER} !important;
    gap: 4px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {BG} !important;
    color: {TEXT2} !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 10px 22px !important;
    border-radius: 8px 8px 0 0 !important;
}}
.stTabs [aria-selected="true"] {{
    color: #e8500a !important;
    border-bottom: 3px solid #e8500a !important;
    background-color: {"#1e293b" if dark else "#fff5f0"} !important;
}}
[data-testid="stTabsContent"],
[data-testid="stTabsContent"] > div {{ background-color: {BG} !important; }}

[data-testid="stExpander"] {{
    background-color: {BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {{
    color: {TEXT} !important; font-weight: 600 !important;
}}
[data-testid="stExpander"] > div > div {{
    background-color: {BG} !important; color: {TEXT} !important;
}}

[data-testid="stFileUploader"] {{
    background-color: {BG2} !important;
    border: 2px dashed {BG3} !important;
    border-radius: 10px !important; padding: 16px !important;
}}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{ color: {TEXT} !important; }}

[data-testid="stDataFrame"] {{ background-color: {BG} !important; }}
.dvn-scroller {{ background-color: {BG} !important; }}
hr {{ border-color: {BORDER} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {TEXT2} !important; }}

.block-container {{ padding: 1.6rem 2.8rem 2rem 2.8rem !important; max-width: 1300px; }}

/* topbar */
.ms-bar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0 16px 0; border-bottom: 2px solid {BORDER}; margin-bottom: 28px;
    background: {BG} !important;
    animation: fadeIn 0.5s ease both;
}}
.ms-wordmark {{ font-size: 1.55rem; font-weight: 900; color: {TEXT} !important; letter-spacing: -0.4px; }}
.ms-tagline {{ font-size: 0.78rem; color: {TEXT2} !important; margin-top: 2px; font-style: italic; }}
.ms-pill-ok   {{ font-size: 0.73rem; font-weight: 700; color: #166534 !important;
                background: #dcfce7 !important; border-radius: 99px; padding: 4px 13px; }}
.ms-pill-warn {{ font-size: 0.73rem; font-weight: 700; color: #92400e !important;
                background: #fef3c7 !important; border-radius: 99px; padding: 4px 13px; }}
.ms-wordmark em {{
    background: linear-gradient(90deg, #e8500a 25%, #f97316 50%, #e8500a 75%);
    background-size: 300% 100%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 2.5s linear infinite; font-style: normal;
}}

/* hero */
.ms-hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 14px; padding: 44px 48px; margin-bottom: 28px;
    animation: fadeInUp 0.6s cubic-bezier(.22,.68,0,1.2) both,
               heroGlow 4s ease-in-out 0.8s infinite;
}}
.ms-hero, .ms-hero * {{ color: #ffffff !important; }}
.ms-hero-steps {{ display: flex; gap: 12px; flex-wrap: wrap; }}
.ms-step {{
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
    border-radius: 8px; padding: 10px 16px; font-size: 0.85rem;
    color: #ffffff !important; display: flex; align-items: center; gap: 8px;
}}
.ms-hero-steps .ms-step:nth-child(1) {{ animation: slideInLeft 0.45s .15s both; }}
.ms-hero-steps .ms-step:nth-child(2) {{ animation: slideInLeft 0.45s .25s both; }}
.ms-hero-steps .ms-step:nth-child(3) {{ animation: slideInLeft 0.45s .35s both; }}
.ms-hero-steps .ms-step:nth-child(4) {{ animation: slideInLeft 0.45s .45s both; }}

/* decision cards */
.ms-card {{ border-radius: 12px; padding: 20px 24px; margin-bottom: 0; }}
.ms-accept {{ background: {"#052e16" if dark else "#f0fdf4"} !important; border: 2px solid #16a34a;
              animation: scaleIn 0.45s cubic-bezier(.22,.68,0,1.2) both,
                         pulseGreen 2.4s ease-in-out 0.6s infinite; }}
.ms-reject {{ background: {"#450a0a" if dark else "#fef2f2"} !important; border: 2px solid #dc2626;
              animation: scaleIn 0.45s cubic-bezier(.22,.68,0,1.2) both,
                         pulseRed 2.4s ease-in-out 0.6s infinite; }}
.ms-review {{ background: {"#422006" if dark else "#fffbeb"} !important; border: 2px solid #d97706;
              animation: scaleIn 0.45s cubic-bezier(.22,.68,0,1.2) both,
                         pulseAmber 2.4s ease-in-out 0.6s infinite; }}
.ms-card, .ms-card div, .ms-card span, .ms-card p {{ color: {TEXT} !important; }}
.ms-label  {{ font-size: 0.68rem; font-weight: 800; text-transform: uppercase;
              letter-spacing: .8px; margin-bottom: 4px; display: block; }}
.ms-accept .ms-label {{ color: #16a34a !important; }}
.ms-reject .ms-label {{ color: #dc2626 !important; }}
.ms-review .ms-label {{ color: #d97706 !important; }}
.ms-decision {{ font-size: 2rem; font-weight: 900; margin: 0; display: block; }}
.ms-accept .ms-decision {{ color: #15803d !important; }}
.ms-reject .ms-decision {{ color: #b91c1c !important; }}
.ms-review .ms-decision {{ color: #b45309 !important; }}
.ms-sub {{ font-size: 0.84rem; color: {TEXT2} !important; margin-top: 6px; line-height: 1.5; display: block; }}

/* alert banners */
.ms-domain-warn {{
    background: {"#431407" if dark else "#fff7ed"} !important; border-left: 5px solid #ea580c;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 14px 0; font-size: 0.88rem;
}}
.ms-domain-warn, .ms-domain-warn * {{ color: {"#fed7aa" if dark else "#431407"} !important; }}
.ms-model-note {{
    background: {"#172554" if dark else "#eff6ff"} !important; border-left: 5px solid #2563eb;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 14px 0; font-size: 0.85rem;
}}
.ms-model-note, .ms-model-note * {{ color: {"#bfdbfe" if dark else "#1e3a5f"} !important; }}

/* section headings */
.ms-section {{
    font-size: 0.9rem; font-weight: 800; color: {TEXT} !important;
    text-transform: uppercase; letter-spacing: .6px;
    border-left: 3px solid #e8500a; padding-left: 10px;
    margin: 22px 0 12px 0; display: block; background: transparent !important;
    animation: slideInLeft 0.4s cubic-bezier(.22,.68,0,1.2) both;
}}

/* tags */
.tag-real  {{ display:inline-block; font-size:0.67rem; font-weight:700;
             text-transform:uppercase; background:#dcfce7 !important; color:#166534 !important;
             border-radius:4px; padding:2px 7px; margin-left:6px; }}
.tag-synth {{ display:inline-block; font-size:0.67rem; font-weight:700;
             text-transform:uppercase; background:#fef9c3 !important; color:#713f12 !important;
             border-radius:4px; padding:2px 7px; margin-left:6px; }}

/* action cards */
.ms-action-card {{ border-radius: 10px; padding: 16px 18px; margin-bottom: 10px; border-left: 5px solid; }}
.ms-action-immediate   {{ background:{"#450a0a" if dark else "#fef2f2"} !important; border-color:#dc2626;
                          animation: fadeInUp 0.5s .10s both; }}
.ms-action-investigate {{ background:{"#172554" if dark else "#eff6ff"} !important; border-color:#2563eb;
                          animation: fadeInUp 0.5s .22s both; }}
.ms-action-correct     {{ background:{"#052e16" if dark else "#f0fdf4"} !important; border-color:#16a34a;
                          animation: fadeInUp 0.5s .34s both; }}
.ms-action-verify      {{ background:{"#2e1065" if dark else "#faf5ff"} !important; border-color:#9333ea;
                          animation: fadeInUp 0.5s .46s both; }}
.ms-action-card, .ms-action-card div, .ms-action-card span, .ms-action-card p {{ color: {TEXT} !important; }}
.ms-action-title {{ font-size:0.72rem; font-weight:800; text-transform:uppercase;
                    letter-spacing:.7px; margin-bottom:8px; display:block; }}
.ms-action-immediate  .ms-action-title {{ color:#dc2626 !important; }}
.ms-action-investigate .ms-action-title {{ color:#2563eb !important; }}
.ms-action-correct    .ms-action-title {{ color:#16a34a !important; }}
.ms-action-verify     .ms-action-title {{ color:#9333ea !important; }}

/* pvar */
.ms-pvar {{
    background: {BG2} !important; border: 1px solid {BORDER};
    border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;
    animation: scaleIn 0.4s cubic-bezier(.22,.68,0,1.2) both 0.2s;
}}
.ms-pvar, .ms-pvar * {{ color: {TEXT} !important; }}
.ms-pvar-label {{ font-size:0.73rem; font-weight:700; color:{TEXT2} !important;
                 text-transform:uppercase; letter-spacing:.5px; display: block; }}
.ms-pvar-value {{ font-size:1.8rem; font-weight:900; color:{TEXT} !important; display: block; }}
.ms-pvar-status-ok   {{ font-size:0.78rem; font-weight:700; color:#16a34a !important; display: block; }}
.ms-pvar-status-warn {{ font-size:0.78rem; font-weight:700; color:#d97706 !important; display: block; }}
.ms-pvar-status-high {{ font-size:0.78rem; font-weight:700; color:#dc2626 !important; display: block; }}

/* confidence bar */
.ms-conf-wrap {{ background:{BG3}; border-radius:99px; height:10px; margin-top:4px; overflow:hidden; }}
.ms-conf-bar  {{ height:10px; border-radius:99px;
                 animation: confFill 1.1s cubic-bezier(.22,.68,0,1.2) both 0.3s; }}

/* scanning loader */
.ms-scan-wrap {{
    position: relative; width: 100%; height: 8px;
    background: {BG3}; border-radius: 99px; overflow: hidden; margin: 12px 0;
}}
.ms-scan-bar {{
    position: absolute; top: 0; left: -40%; width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, #e8500a, transparent);
    border-radius: 99px; animation: scanBar 1.2s linear infinite;
}}

/* progress breadcrumb */
.ms-prog-step {{
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.82rem; font-weight: 600; color: #16a34a;
    animation: progressPop 0.4s cubic-bezier(.22,.68,0,1.2) both;
}}
.ms-prog-arrow {{ color: {TEXT2}; font-size: 0.75rem; }}

/* metric count-up */
[data-testid="stMetricValue"] div {{ animation: countUp 0.5s cubic-bezier(.22,.68,0,1.2) both; }}

/* history table rows */
.ms-hist-row {{
    display:flex; gap:10px; align-items:center;
    padding:9px 14px; border-radius:8px; margin-bottom:6px;
    background:{BG2}; border:1px solid {BORDER};
    animation: rowIn 0.4s cubic-bezier(.22,.68,0,1.2) both;
    font-size:0.84rem; color:{TEXT};
}}
.ms-hist-row:hover {{ background:{BG3}; }}

/* reference gallery card */
.ms-gal-card {{
    background:{BG2}; border:1px solid {BORDER}; border-radius:12px;
    padding:12px; text-align:center;
    animation: scaleIn 0.4s cubic-bezier(.22,.68,0,1.2) both;
    transition: transform 0.2s, box-shadow 0.2s;
}}
.ms-gal-card:hover {{ transform: translateY(-3px);
                      box-shadow: 0 8px 24px rgba(0,0,0,0.15); }}
.ms-gal-label {{ font-size:0.8rem; font-weight:700; margin-top:8px; color:{TEXT}; }}

/* batch result row */
.ms-batch-accept {{ color:#16a34a; font-weight:700; }}
.ms-batch-reject {{ color:#dc2626; font-weight:700; }}
.ms-batch-review {{ color:#d97706; font-weight:700; }}

/* footer */
.ms-footer {{
    font-size:0.72rem; color:{TEXT2} !important; text-align:center;
    margin-top:48px; padding-top:14px; border-top:1px solid {BORDER};
    background:{BG} !important;
    animation: fadeIn 0.6s ease 0.3s both;
}}
.ms-footer strong {{ color:{TEXT} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ResNet18 model…")
def _load_predictor():
    from app.vision.predictor import DefectPredictor
    p = DefectPredictor()
    p.load_model()
    return p


@st.cache_data(show_spinner=False)
def _load_synthetic():
    from app.data.synthetic import (
        load_or_generate,
        compute_station_metrics,
        compute_economics_summary,
    )
    prod_df, econ_df = load_or_generate()
    station_metrics  = compute_station_metrics(prod_df)
    econ_summary     = compute_economics_summary(econ_df)
    return prod_df, econ_df, station_metrics, econ_summary


model_ready = settings.model_weights_path.exists()

# ─────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────
KB: dict[str, dict] = {
    "crack": {
        "causes": [
            "Excessive mechanical stress or vibration during processing",
            "Process parameter deviation — cycle time, feed rate, machine speed",
            "Tool wear or fixture misalignment introducing uneven force",
            "Post-changeover instability (quality transiently degrades after setup change)",
            "Material batch variation in hardness or brittleness",
        ],
        "immediate": [
            "HOLD this unit — do not release to the next process step",
            "Hold adjacent units from the same production batch",
            "Log defect against current batch ID and station",
            "Notify the responsible process engineer immediately",
        ],
        "investigate": [
            "Measure vibration levels at the responsible station — compare to approved baseline",
            "Check cycle-time deviation from standard operating range",
            "Inspect tooling condition for wear, chips, or misalignment",
            "Verify fixture clamping force and alignment",
            "Review defect trend across the last 3–5 batches — increasing rate signals drift",
        ],
        "correct": [
            "Restore process parameters to approved operating range if deviation is confirmed",
            "Recondition or replace worn tooling if confirmed as a contributing factor",
            "Implement vibration dampening or isolation if elevated vibration is confirmed",
            "Review and tighten fixturing if mechanical stress concentration is identified",
            "Increase post-changeover inspection frequency until stability is confirmed",
        ],
        "verify": [
            "Run a 10–20 unit verification batch after corrective action",
            "Inspect all verification units for crack indicators",
            "Compare crack defect rate: pre-correction vs verification batch",
            "Document corrective action and measured outcome in quality record",
        ],
        "monitor": [
            "Crack defect rate by station and batch (target: return to baseline)",
            "Vibration readings — alert if above approved threshold",
            "Cycle-time deviation — flag if outside ±10% of standard",
        ],
        "pvar": "vibration_mms", "pvar_label": "Vibration",
        "pvar_unit": "mm/s", "pvar_warn": 0.7, "pvar_high": 1.0,
    },
    "hole": {
        "causes": [
            "Intentional product geometry (designed holes, slots, countersinks) — verify against drawing first",
            "Tooling wear or breakage causing incorrect material removal",
            "Incorrect or drifted pressure/feed parameter",
            "Fixture misalignment causing off-location features",
            "Material thickness or hardness variation",
        ],
        "immediate": [
            "COMPARE this unit against the approved product drawing or reference image",
            "Verify whether the observed feature is intentional product geometry",
            "If geometry is confirmed defective: HOLD the batch",
            "Do NOT reject automatically without human visual confirmation",
        ],
        "investigate": [
            "Check approved product drawing — is this feature expected?",
            "Inspect tooling for wear, breakage, or misalignment",
            "Verify pressure and feed rate against approved operating range",
            "Review station setup records — any changes to tooling or settings recently?",
            "Check adjacent units — consistent pattern = tooling issue, isolated = material issue",
        ],
        "correct": [
            "If intentional geometry: no action — update reference/inspection process",
            "If defective: restore pressure/feed settings to approved range",
            "Replace or realign worn/broken tooling",
            "Isolate and re-inspect full affected batch if systematic error is suspected",
        ],
        "verify": [
            "Run a verification batch after confirmed corrective action",
            "Compare each verification unit against reference drawing",
            "Confirm defect rate has returned to baseline",
        ],
        "monitor": [
            "Hole/feature defect rate by station",
            "Tooling wear interval — log tool changes and quality outcomes",
            "Pressure/feed deviation — alert if outside approved range",
        ],
        "pvar": "pressure_bar", "pvar_label": "Pressure",
        "pvar_unit": "bar", "pvar_warn": 6.0, "pvar_high": 7.5,
    },
    "rust": {
        "causes": [
            "Moisture or humidity exposure during storage or production",
            "Surface treatment / protective coating failure or insufficient coverage",
            "Extended storage beyond approved shelf life",
            "Condensation from temperature fluctuation in storage area",
            "Handling-induced coating damage exposing bare metal",
        ],
        "immediate": [
            "HOLD this unit and the associated batch pending verification",
            "Move affected units to a dry, controlled environment immediately",
            "Log defect against batch ID and station",
            "Notify quality and materials engineering",
        ],
        "investigate": [
            "Review surface treatment / coating process records for the affected batch",
            "Check humidity and temperature logs for production and storage areas",
            "Verify storage duration — has the material exceeded approved shelf life?",
            "Inspect material handling records for coating damage events",
            "Check environmental controls (dehumidifiers, seals) for failures",
        ],
        "correct": [
            "Quarantine affected batch for 100% inspection if rust is widespread",
            "Correct surface treatment or coating process if failure is confirmed",
            "Improve environmental controls — humidity target: < 50% RH",
            "Update FIFO rotation procedures if shelf-life exceedance is identified",
            "Retrain handling personnel if mechanical coating damage is root cause",
        ],
        "verify": [
            "Inspect a sample batch produced after corrective action",
            "Measure surface coating thickness and continuity",
            "Check humidity logs for 48 hours after environmental correction",
            "Compare rust incidence rate: pre- vs post-correction",
        ],
        "monitor": [
            "Rust defect rate by batch and storage duration",
            "Humidity and temperature in storage areas — daily log",
            "Surface treatment process records — track yield per coating run",
        ],
        "pvar": "humidity_pct", "pvar_label": "Humidity",
        "pvar_unit": "%RH", "pvar_warn": 60.0, "pvar_high": 75.0,
    },
    "scratch": {
        "causes": [
            "Damaged or rough contact surface on conveyor, fixture, or tooling",
            "Improper handling during transfer between process steps",
            "Part-to-part contact during transport or packaging",
            "Excessive vibration causing relative movement between part and surface",
            "Recent maintenance or tooling change introducing a new abrasive contact edge",
        ],
        "immediate": [
            "HOLD this unit pending verification",
            "Visually inspect the handling path for the last process step",
            "Determine whether the scratch pattern is consistent (fixed contact point) or random",
            "Log defect against current batch and station",
        ],
        "investigate": [
            "Inspect all contact surfaces: conveyor belts, fixtures, transfer guides, packaging",
            "Note scratch location and orientation — consistent = fixed contact point",
            "Review vibration data — excessive vibration can cause part movement and abrasion",
            "Check for recent maintenance or tool changes that introduced new contact edges",
            "Review scratch rate trend — sudden increase indicates a new abrasive contact point",
        ],
        "correct": [
            "Remove or repair the identified abrasive / damaged contact surface",
            "Add protective padding, separators, or liners at the identified contact point",
            "Replace damaged conveyor components or fixture inserts",
            "Adjust handling procedures to minimise direct part contact",
            "Verify tooling surface finish is within approved roughness specification",
        ],
        "verify": [
            "Run a 10–20 unit verification batch after removing/repairing the contact point",
            "Inspect verification units for scratch indicators",
            "Confirm scratch rate has returned to baseline",
            "Check repaired surface condition after first production run",
        ],
        "monitor": [
            "Scratch defect rate by station — alert if above baseline",
            "Periodic inspection of high-risk contact surfaces",
            "Vibration readings at the responsible station",
        ],
        "pvar": "vibration_mms", "pvar_label": "Vibration",
        "pvar_unit": "mm/s", "pvar_warn": 0.7, "pvar_high": 1.0,
    },
    "normal": {
        "causes": [],
        "immediate":    ["No hold required — continue production"],
        "investigate":  ["Continue standard sampling inspection per your quality plan"],
        "correct":      ["No corrective action required"],
        "verify":       ["Maintain standard monitoring frequency"],
        "monitor":      ["Watch for gradual increase in defect rate over time"],
        "pvar": None, "pvar_label": None, "pvar_unit": None,
        "pvar_warn": None, "pvar_high": None,
    },
}

CLASS_COLORS = {
    "normal":  "#16a34a",
    "crack":   "#dc2626",
    "hole":    "#f97316",
    "rust":    "#b45309",
    "scratch": "#2563eb",
}

GRADCAM_PATHS = {
    cls: PROJECT_ROOT / "reports" / "gradcam" / f"gradcam_{cls}.png"
    for cls in ["crack", "hole", "normal", "rust", "scratch"]
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def _raw_decision(predicted_class: str, confidence: float, threshold: float | None = None) -> str:
    high = threshold or st.session_state.conf_threshold
    low  = float(settings.uncertainty_threshold)
    if confidence < low:
        return "REVIEW"
    if predicted_class == "normal":
        return "ACCEPT" if confidence >= high else "REVIEW"
    return "REJECT" if confidence >= high else "REVIEW"


def assess_reliability(
    probabilities: dict[str, float],
    confidence: float,
    predicted_class: str,
    tensor: Any,
) -> dict:
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    top1_cls, top1_val = sorted_probs[0]
    top2_val = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin   = top1_val - top2_val
    thr      = st.session_state.conf_threshold
    low_conf   = confidence < thr
    low_margin = margin < 0.30

    domain_shift = False
    domain_note  = ""
    if tensor is not None:
        try:
            arr   = tensor.squeeze(0).cpu().numpy()
            pmean = float(arr.mean())
            pstd  = float(arr.std())
            if pmean > 0.6 or pmean < -1.5:
                domain_shift = True
                domain_note = (
                    f"Image brightness (normalised mean {pmean:.2f}) is outside the "
                    "range typical of training images. Prediction may be unreliable."
                )
            elif pstd < 0.05:
                domain_shift = True
                domain_note = (
                    "Image has very low contrast (near-uniform pixel values). "
                    "The model was trained on textured industrial surfaces — "
                    "this image may not match that distribution."
                )
        except Exception:
            pass

    confusion_note = ""
    if predicted_class == "hole" and probabilities.get("rust", 0) > 0.15:
        confusion_note = (
            "⚠️ Note: The model's primary confusion pattern is rust→hole "
            "(14 / 16 test errors). If this part is metallic and shows "
            "discolouration, rust is a more likely explanation — verify visually."
        )

    flags = []
    if low_conf:
        flags.append(
            f"Confidence {confidence:.1%} is below threshold ({thr:.0%}). Manual inspection required."
        )
    if low_margin:
        top2_cls = sorted_probs[1][0] if len(sorted_probs) > 1 else "?"
        flags.append(
            f"Narrow margin ({margin:.1%}) — model not clearly favouring "
            f"{top1_cls.capitalize()} over {top2_cls.capitalize()}."
        )
    if domain_shift:
        flags.append(domain_note)
    if confusion_note:
        flags.append(confusion_note)

    reliable     = not low_conf and not low_margin and not domain_shift
    force_review = (not reliable) and (predicted_class != "normal")
    raw_decision = _raw_decision(predicted_class, confidence)
    display_decision = "REVIEW" if (force_review and raw_decision == "REJECT") else raw_decision

    return {
        "display_decision": display_decision,
        "raw_decision":     raw_decision,
        "reliable":         reliable,
        "flags":            flags,
        "margin":           margin,
        "low_margin":       low_margin,
        "domain_shift":     domain_shift,
        "domain_note":      domain_note,
        "force_review":     force_review,
        "confusion_note":   confusion_note,
    }


def get_process_row(prod_df: pd.DataFrame, predicted_class: str, filename: str) -> pd.Series:
    seed = int(hashlib.md5(filename.encode()).hexdigest(), 16) % (2 ** 31)
    rng  = np.random.default_rng(seed)
    pool = prod_df[prod_df["true_defect_label"] == predicted_class]
    if len(pool) == 0:
        pool = prod_df
    return pool.iloc[int(rng.integers(0, len(pool)))]


def _chart(fig: go.Figure, key: str, height: int = 280) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=36, b=12),
        font=dict(family="Inter, system-ui, sans-serif", size=12,
                  color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def _run_single_inference(file_bytes: bytes, filename: str, predictor) -> dict:
    """Run inference on a single image bytes blob. Returns a result dict."""
    tmp_path = None
    tensor   = None
    try:
        suffix = Path(filename).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(file_bytes)
            tmp_path = f.name
        pred = predictor.predict(tmp_path)
        from app.vision.preprocessing import preprocess_single_image
        tensor = preprocess_single_image(tmp_path)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    rel = assess_reliability(
        probabilities=pred.probabilities,
        confidence=pred.confidence,
        predicted_class=pred.predicted_class,
        tensor=tensor,
    )
    return {
        "pred":   pred,
        "tensor": tensor,
        "rel":    rel,
    }


def _make_csv_report(pred, rel: dict, insp_id: str) -> bytes:
    """Build a CSV bytes report for a single inspection."""
    rows = [
        ["Inspection ID", insp_id],
        ["Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Predicted Class", pred.predicted_class],
        ["Confidence", f"{pred.confidence:.4f}"],
        ["Decision", rel["display_decision"]],
        ["Prediction Margin", f"{rel['margin']:.4f}"],
        ["Reliable", str(rel["reliable"])],
        ["Domain Shift", str(rel["domain_shift"])],
        ["Flags", " | ".join(rel["flags"]) if rel["flags"] else "None"],
        [],
        ["Class Probabilities"],
    ]
    for cls, prob in sorted(pred.probabilities.items(), key=lambda x: -x[1]):
        rows.append([cls, f"{prob:.6f}"])

    buf = io.StringIO()
    for row in rows:
        buf.write(",".join(str(c) for c in row) + "\n")
    return buf.getvalue().encode()


# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
pill = (
    '<span class="ms-pill-ok">● Model Ready · ResNet18 · 99.11% test accuracy</span>'
    if model_ready else
    '<span class="ms-pill-warn">● Model not found — run training first</span>'
)
st.markdown(f"""
<div class="ms-bar">
  <div style="display:flex; align-items:center; gap:12px;">
    <svg width="36" height="36" viewBox="0 0 52 52" fill="none">
      <rect width="52" height="52" rx="10" fill="#e8500a"/>
      <rect x="8" y="30" width="6" height="14" fill="white"/>
      <rect x="16" y="24" width="6" height="20" fill="white"/>
      <rect x="24" y="18" width="6" height="26" fill="white"/>
      <rect x="32" y="22" width="6" height="22" fill="white"/>
      <rect x="40" y="28" width="4" height="16" fill="white"/>
      <circle cx="11" cy="26" r="2.5" fill="#fbbf24"/>
      <circle cx="19" cy="20" r="2.5" fill="#fbbf24"/>
      <circle cx="27" cy="14" r="2.5" fill="#fbbf24"/>
      <circle cx="35" cy="18" r="2.5" fill="#fbbf24"/>
      <circle cx="42" cy="24" r="2.5" fill="#fbbf24"/>
      <polyline points="11,26 19,20 27,14 35,18 42,24" stroke="#fbbf24" stroke-width="1.5" fill="none"/>
    </svg>
    <div>
      <div class="ms-wordmark">Manu<em>Sense</em></div>
      <div class="ms-tagline">AI-Powered Manufacturing Quality &amp; Root-Cause Intelligence</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    {pill}
    <span style="font-size:0.72rem;color:{TEXT2};background:{BG2};
                 border:1px solid {BORDER};border-radius:99px;padding:4px 12px;">
      Threshold: {st.session_state.conf_threshold:.0%}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_inspect, tab_batch, tab_history, tab_gallery, tab_whatif, tab_model = st.tabs([
    "🔍  Inspection",
    "📦  Batch Upload",
    "📋  History",
    "🖼️  Reference Gallery",
    "🧪  What-If Explorer",
    "📊  Model Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_inspect:

    st.markdown(f"""
<div class="ms-hero">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
    <svg width="56" height="56" viewBox="0 0 56 56" fill="none"
         style="animation: scaleIn 0.5s cubic-bezier(.22,.68,0,1.2) both 0.1s;">
      <rect width="56" height="56" rx="14" fill="#e8500a"/>
      <rect x="9" y="32" width="7" height="16" rx="2" fill="white"/>
      <rect x="18" y="26" width="7" height="22" rx="2" fill="white"/>
      <rect x="27" y="19" width="7" height="29" rx="2" fill="white"/>
      <rect x="36" y="23" width="7" height="25" rx="2" fill="white"/>
      <circle cx="12" cy="29" r="2.5" fill="#fbbf24"/>
      <circle cx="21" cy="23" r="2.5" fill="#fbbf24"/>
      <circle cx="30" cy="16" r="2.5" fill="#fbbf24"/>
      <circle cx="39" cy="20" r="2.5" fill="#fbbf24"/>
      <polyline points="12,29 21,23 30,16 39,20" stroke="#fbbf24" stroke-width="2" fill="none"/>
    </svg>
    <div style="animation: fadeInUp 0.5s cubic-bezier(.22,.68,0,1.2) both 0.2s;">
      <div style="font-size:2rem;font-weight:900;color:#ffffff;letter-spacing:-0.5px;line-height:1;">
        Manu<span style="color:#f97316;">Sense</span>
      </div>
      <div style="font-size:0.8rem;color:#94a3b8;margin-top:4px;">
        AI-Powered Manufacturing Quality &amp; Root-Cause Intelligence
      </div>
    </div>
  </div>
  <p style="color:#cbd5e1;font-size:1rem;margin:0 0 20px 0;line-height:1.6;
            animation: fadeIn 0.5s ease both 0.35s;">
    Upload a surface inspection image. ManuSense classifies the defect,
    explains the model decision, identifies root-cause factors, and generates
    a prioritised corrective-action plan.
  </p>
  <div class="ms-hero-steps">
    <div class="ms-step"><span class="ms-step-num">1</span> Upload image</div>
    <div class="ms-step"><span class="ms-step-num">2</span> ResNet18 classifies</div>
    <div class="ms-step"><span class="ms-step-num">3</span> Grad-CAM explains</div>
    <div class="ms-step"><span class="ms-step-num">4</span> Root cause &amp; action plan</div>
  </div>
</div>
""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a surface inspection image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
        help="PNG · JPG · BMP · TIFF · Grayscale industrial surface images recommended",
        key="single_uploader",
    )

    if uploaded is None:
        st.markdown(f"""
<div style="text-align:center; padding:40px 32px; color:{TEXT2}; font-size:0.92rem;
            background:{BG2}; border-radius:12px; border:2px dashed {BG3};
            margin-top:8px; animation: fadeInUp 0.5s cubic-bezier(.22,.68,0,1.2) both 0.2s;">
  <div style="font-size:2.2rem; margin-bottom:12px; animation: fadeIn 0.4s ease 0.4s both;">📤</div>
  <div style="font-weight:700; color:{TEXT}; font-size:1rem; margin-bottom:6px;">
    Drop your inspection image here
  </div>
  <div>or click <strong>Browse files</strong> above to upload</div>
  <div style="font-size:0.78rem; color:{TEXT2}; margin-top:10px;">
    Supported: PNG · JPG · JPEG · BMP · TIFF &nbsp;|&nbsp; Best: 256×256 grayscale surface texture
  </div>
</div>
""", unsafe_allow_html=True)

    else:  # ── image uploaded — run full pipeline ────────────────────────────

        if not model_ready:
            st.error("**Model weights not found.** Run `python -m app.vision.train` to train the model first.")
        else:
            try:
                predictor = _load_predictor()
            except Exception as exc:
                st.error(f"Could not load model: {exc}")
                predictor = None

            if predictor is not None:
                prod_df = econ_df = sm_df = econ_summary = None
                try:
                    prod_df, econ_df, sm_df, econ_summary = _load_synthetic()
                except Exception as exc:
                    log.warning("Synthetic data unavailable: %s", exc)
                    econ_summary = {}

                # scanning loader
                scan_ph = st.empty()
                scan_ph.markdown(f"""
<div style="padding:14px 18px; background:{BG2}; border-radius:10px;
            border:1px solid {BORDER}; margin-bottom:12px;">
  <div style="font-size:0.88rem; font-weight:600; color:{TEXT}; margin-bottom:8px;">
    🔬 Analysing image with ResNet18…
  </div>
  <div class="ms-scan-wrap"><div class="ms-scan-bar"></div></div>
  <div style="font-size:0.75rem; color:{TEXT2}; margin-top:6px;">
    Running forward pass · computing Grad-CAM · evaluating reliability…
  </div>
</div>
""", unsafe_allow_html=True)

                inference_ok = False
                with st.spinner(""):
                    try:
                        result = _run_single_inference(uploaded.getvalue(), uploaded.name, predictor)
                        pred   = result["pred"]
                        tensor = result["tensor"]
                        rel    = result["rel"]
                        inference_ok = True
                    except Exception as exc:
                        scan_ph.empty()
                        st.error(f"Inference failed: {exc}")

                scan_ph.empty()

                if inference_ok:
                    # Grad-CAM
                    gc_overlay = None
                    gc_error   = ""
                    try:
                        from app.vision.explainability import explain_prediction
                        gc_result  = explain_prediction(model=predictor._model, tensor=tensor)
                        gc_overlay = gc_result["overlay"]
                    except Exception as exc:
                        gc_error = str(exc)
                        log.warning("Grad-CAM failed: %s", exc)

                display_dec   = rel["display_decision"]
                predicted_cls = pred.predicted_class
                confidence    = pred.confidence
                kb            = KB.get(predicted_cls, KB["normal"])

                if display_dec == "REVIEW" and predicted_cls != "normal":
                    action_immediate  = [
                        "Do NOT automatically reject — human confirmation is required",
                        "Route this unit to the manual inspection station",
                        "Re-inspect under controlled lighting against a reference image",
                        "Compare with approved product drawing if circular features are present",
                    ]
                    action_investigate = [
                        "Verify predicted class visually before any further action",
                        "Check if observed feature matches intentional product geometry",
                        "Consult reference images for this product variant",
                    ]
                    action_correct = ["No corrective action until defect is visually confirmed by a human inspector"]
                    action_verify  = kb["verify"]
                else:
                    action_immediate   = kb["immediate"]
                    action_investigate = kb["investigate"]
                    action_correct     = kb["correct"]
                    action_verify      = kb["verify"]

                short_id = hashlib.md5(uploaded.name.encode()).hexdigest()[:6].upper()
                insp_id  = f"MS-{short_id}"

                # ── TASK 3: append to history ─────────────────────────────────────────────
                if not st.session_state.history or st.session_state.history[-1]["insp_id"] != insp_id:
                    st.session_state.history.append({
                        "insp_id":   insp_id,
                        "filename":  uploaded.name,
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "class":     predicted_cls,
                        "confidence": confidence,
                        "decision":  display_dec,
                        "margin":    rel["margin"],
                    })
                    st.session_state.session_counts["total"] += 1
                    st.session_state.session_counts[display_dec] = (
                        st.session_state.session_counts.get(display_dec, 0) + 1
                    )

                # progress breadcrumb
                breadcrumb_steps = [
                    ("✓ Image loaded", "0.05s"),
                    ("✓ Model run", "0.15s"),
                    (f'✓ Grad-CAM{"" if gc_overlay is not None else " (unavail.)"}', "0.25s"),
                    ("✓ Root-cause analysis", "0.35s"),
                    ("✓ Report ready", "0.45s"),
                ]
                bc_html = ""
                for i, (label, delay) in enumerate(breadcrumb_steps):
                    bc_html += f'<span class="ms-prog-step" style="animation-delay:{delay};">{label}</span>'
                    if i < len(breadcrumb_steps) - 1:
                        d2 = f"{float(delay[:-1]) + 0.05:.2f}s"
                        bc_html += f'<span class="ms-prog-arrow" style="animation:fadeIn 0.4s {d2} both;">›</span>'

                st.markdown(f"""
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:18px;font-size:0.8rem;
                        flex-wrap:wrap;background:{BG2};border-radius:8px;padding:10px 14px;
                        border:1px solid {BORDER};animation:fadeIn 0.4s ease both;">
              {bc_html}
              <span style="margin-left:auto;color:{TEXT2};font-size:0.75rem;
                           animation:fadeIn 0.4s 0.55s both;display:inline-block;">
                ID: <strong style="color:{TEXT};">{insp_id}</strong>
              </span>
            </div>
            """, unsafe_allow_html=True)

                st.divider()

                # ── IMAGE + DECISION CARD ─────────────────────────────────────────────────
                col_img, col_dec = st.columns([1, 1], gap="large")

                with col_img:
                    st.markdown('<div class="ms-section anim-slideLeft">Uploaded Image</div>', unsafe_allow_html=True)
                    st.image(uploaded, use_container_width=True)

                with col_dec:
                    st.markdown('<div class="ms-section anim-slideRight">Inspection Decision</div>', unsafe_allow_html=True)

                    dec_icon = {"ACCEPT": "✅", "REJECT": "❌", "REVIEW": "🔍"}.get(display_dec, "⚠️")
                    card_cls = {"ACCEPT": "ms-accept", "REJECT": "ms-reject"}.get(display_dec, "ms-review")

                    if display_dec == "ACCEPT":
                        sub_text = "No defect detected — part passes visual inspection."
                    elif display_dec == "REJECT":
                        sub_text = (
                            f"<strong>{predicted_cls.capitalize()}</strong> defect detected with "
                            f"{confidence:.1%} confidence. Remove from production flow."
                        )
                    else:
                        sub_text = (
                            f"Model predicted <strong>{predicted_cls.capitalize()}</strong> "
                            f"({confidence:.1%}) — confidence or margin insufficient for auto-rejection. "
                            "Manual inspection required."
                        )

                    st.markdown(f"""
            <div class="ms-card {card_cls}">
              <div class="ms-label">Inspection Decision</div>
              <div class="ms-decision">{dec_icon} {display_dec}</div>
              <div class="ms-sub">{sub_text}</div>
            </div>
            """, unsafe_allow_html=True)

                    st.markdown("")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Class",   predicted_cls.capitalize())
                    m2.metric("Confidence",        f"{confidence:.1%}")
                    m3.metric("Prediction Margin", f"{rel['margin']:.1%}",
                              help="Gap between top-1 and top-2 probabilities. <30% = low margin.")

                    # animated confidence bar
                    conf_pct  = int(confidence * 100)
                    bar_color = "#16a34a" if display_dec == "ACCEPT" else "#dc2626" if display_dec == "REJECT" else "#d97706"
                    st.markdown(f"""
            <div style="margin:4px 0 14px 0;">
              <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:{TEXT2};margin-bottom:4px;">
                <span style="font-weight:600;">Confidence</span>
                <span id="ms-conf-val" style="font-weight:700;color:{bar_color};
                      animation:countUp 0.6s 0.4s both;">{confidence:.1%}</span>
              </div>
              <div class="ms-conf-wrap">
                <div class="ms-conf-bar" style="width:{conf_pct}%;background:{bar_color};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

                    st_components.html(f"""
            <script>
            (function(){{
              var target={confidence*100:.2f},start=0,dur=900,t0=null;
              function step(ts){{
                if(!t0)t0=ts;
                var p=Math.min((ts-t0)/dur,1),e=1-Math.pow(1-p,3),v=(start+(target-start)*e).toFixed(1);
                try{{var s=window.parent.document.getElementById('ms-conf-val');if(s)s.textContent=v+'%';}}catch(e){{}}
                if(p<1)requestAnimationFrame(step);
              }}
              requestAnimationFrame(step);
            }})();
            </script>
            """, height=0)

                    if rel["flags"]:
                        for flag in rel["flags"]:
                            if "rust→hole" in flag or "confusion" in flag.lower():
                                st.info(flag, icon="ℹ️")
                            elif rel["domain_shift"] and flag == rel["domain_note"]:
                                st.markdown(f'<div class="ms-domain-warn">🌐 <strong>Domain notice:</strong> {flag}</div>',
                                            unsafe_allow_html=True)
                            else:
                                st.warning(flag)

                    sorted_probs = sorted(pred.probabilities.items(), key=lambda x: x[1], reverse=True)
                    fig_probs = go.Figure(go.Bar(
                        x=[v for _, v in sorted_probs],
                        y=[c.capitalize() for c, _ in sorted_probs],
                        orientation="h",
                        marker_color=[CLASS_COLORS.get(c, "#6b7280") for c, _ in sorted_probs],
                        text=[f"{v:.1%}" for _, v in sorted_probs],
                        textposition="outside", cliponaxis=False,
                    ))
                    fig_probs.update_layout(
                        xaxis=dict(range=[0, 1.15], tickformat=".0%", title="", showgrid=False),
                        yaxis=dict(title=""), title="Class Probabilities",
                    )
                    _chart(fig_probs, "probs_chart", 220)
                    st.caption("Advisory only — no machine or process control is performed.")

                    # ── TASK 5: CSV export button ─────────────────────────────────────────
                    csv_bytes = _make_csv_report(pred, rel, insp_id)
                    st.download_button(
                        label="⬇️ Download CSV Report",
                        data=csv_bytes,
                        file_name=f"manusense_{insp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                st.divider()

                # ── INNER TABS ────────────────────────────────────────────────────────────
                inner = st.tabs([
                    "🔬 Model Explanation",
                    "🛠️ Root Cause & Corrective Action",
                    "⚙️ Process Evidence",
                    "💰 Cost Impact",
                ])

                with inner[0]:
                    if gc_overlay is not None:
                        c_orig, c_gc = st.columns(2, gap="large")
                        with c_orig:
                            st.markdown('<div class="ms-section">Original Image</div>', unsafe_allow_html=True)
                            st.image(uploaded, use_container_width=True)
                        with c_gc:
                            st.markdown('<div class="ms-section">Grad-CAM Explanation <span class="tag-real">Real model</span></div>',
                                        unsafe_allow_html=True)
                            st.image(gc_overlay, use_container_width=True)
                        st.markdown("""
            <div class="ms-model-note">
              <strong>How to read Grad-CAM:</strong> Warm colours (red/yellow) highlight the regions
              that most influenced the model's prediction. Cool colours had low influence.<br><br>
              <strong>Important:</strong> This shows <em>where the model looked</em>, not a ground-truth
              defect mask. The dataset has no bounding-box annotations.
            </div>
            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Grad-CAM could not be generated: `{gc_error}`\n\nClassification result above is still valid.")

                    with st.expander("Model capability & known limitations", expanded=rel.get("domain_shift", False)):
                        st.markdown("""
            **Validated performance (held-out test set):**
            - Accuracy: **99.11%** — 1,784 / 1,800 correct
            - Dataset: 12,000 images · 5 classes · 70/15/15 stratified split · seed 42
            - Architecture: ResNet18, fine-tuned from ImageNet weights

            **Known error pattern:** Rust → Hole: 14 errors · Rust → Normal: 2 errors · All other pairs: 0

            **Limitations:** Trained on challenge dataset images (256×256 greyscale surface textures).
            External factory images differ from the training distribution.
            """)

                with inner[1]:
                    if display_dec == "ACCEPT":
                        st.success("✅ **No defect detected.** No corrective action required.")
                    else:
                        obs_map = {
                            "crack":   "The model detected a **crack-like surface pattern** — linear or branching discontinuities.",
                            "hole":    "The model detected a **hole-like pattern**. Verify against the approved drawing before concluding a defect.",
                            "rust":    "The model detected a **rust / corrosion pattern** — surface discolouration consistent with oxidation.",
                            "scratch": "The model detected a **scratch-like surface mark** — physical contact damage.",
                        }
                        st.markdown(f"**Model observation:** {obs_map.get(predicted_cls, f'The model detected a {predicted_cls} pattern.')}")

                        if rel["force_review"]:
                            st.markdown('<div class="ms-domain-warn">⚠️ <strong>Reliability warning:</strong> Confidence or margin too low for auto-rejection. Confirm visually before acting.</div>',
                                        unsafe_allow_html=True)

                        if kb["causes"]:
                            st.markdown('<div class="ms-section">Possible Contributing Factors</div>', unsafe_allow_html=True)
                            for cause in kb["causes"]:
                                st.markdown(
                                    f"<div style='font-size:0.92rem;color:{TEXT};padding:3px 0 3px 12px;'>"
                                    f"<span style='color:#e8500a;font-weight:700;'>•</span> {cause}</div>",
                                    unsafe_allow_html=True,
                                )

                        st.markdown('<div class="ms-section anim-fadeInUp">Prioritised Action Plan</div>', unsafe_allow_html=True)

                        for phase_cls, phase_title, phase_color, phase_steps in [
                            ("ms-action-immediate",   "🚨 Phase 1 — Immediate (right now)",              "#dc2626", action_immediate),
                            ("ms-action-investigate", "🔎 Phase 2 — Investigate (before correcting)",    "#2563eb", action_investigate),
                            ("ms-action-correct",     "🔧 Phase 3 — Correct (after confirming cause)",   "#16a34a", action_correct),
                            ("ms-action-verify",      "✅ Phase 4 — Verify &amp; Monitor",               "#9333ea", action_verify),
                        ]:
                            steps_html = "".join(
                                f"<div style='margin:5px 0;font-size:0.9rem;color:{TEXT};"
                                f"animation:slideInLeft 0.4s {0.05*i:.2f}s both;'>"
                                f"<span style='font-weight:800;color:{phase_color};'>▶ Step {i}:</span> {s}</div>"
                                for i, s in enumerate(phase_steps, 1)
                            )
                            st.markdown(f"""
            <div class="ms-action-card {phase_cls}">
              <div class="ms-action-title">{phase_title}</div>
              {steps_html}
            </div>""", unsafe_allow_html=True)

                        if kb["monitor"]:
                            st.markdown('<div class="ms-section">Ongoing Monitoring</div>', unsafe_allow_html=True)
                            for m in kb["monitor"]:
                                st.markdown(
                                    f"<div style='font-size:0.92rem;color:{TEXT};padding:3px 0 3px 12px;'>"
                                    f"<span style='color:#e8500a;font-weight:700;'>•</span> {m}</div>",
                                    unsafe_allow_html=True,
                                )

                with inner[2]:
                    st.markdown(f"""
            <div class="ms-model-note" style="margin-bottom:18px;">
              <strong>Data provenance:</strong> All process records below are
              <strong>synthetic demonstration data</strong>. The challenge dataset provides
              inspection images only. In production, records are joined via
              <em>Inspection ID → MES / production database</em>.
            </div>
            """, unsafe_allow_html=True)

                    if prod_df is not None and sm_df is not None:
                        proc_row = get_process_row(prod_df, predicted_cls, uploaded.name)
                        bn = sm_df.iloc[0]

                        st.markdown('<div class="ms-section">Associated Production Record <span class="tag-synth">Synthetic</span></div>', unsafe_allow_html=True)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Station", str(proc_row["station_id"]))
                        c2.metric("Batch",   str(proc_row["batch_id"]))
                        c3.metric("Shift",   str(proc_row["shift"]))
                        c4.metric("Variant", str(proc_row["product_variant"]))
                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("Cycle Time", f"{proc_row['cycle_time_s']:.1f} s")
                        c6.metric("Downtime",   f"{proc_row['downtime_min']:.1f} min")
                        c7.metric("WIP",        str(int(proc_row["wip"])))
                        c8.metric("Changeover", f"{proc_row['changeover_min']:.1f} min")

                        pvar, pvlabel, pvunit = kb["pvar"], kb["pvar_label"], kb["pvar_unit"]
                        pv_warn, pv_high     = kb["pvar_warn"], kb["pvar_high"]
                        if pvar and pvar in proc_row.index:
                            pv_val = float(proc_row[pvar])
                            st.markdown(f'<div class="ms-section">Key Process Variable: {pvlabel} <span class="tag-synth">Synthetic</span></div>', unsafe_allow_html=True)
                            pvc1, pvc2 = st.columns([1, 2], gap="large")
                            with pvc1:
                                if pv_val >= pv_high:
                                    sc2, st2 = "ms-pvar-status-high", f"⬆ ELEVATED — above {pv_high} {pvunit}"
                                elif pv_val >= pv_warn:
                                    sc2, st2 = "ms-pvar-status-warn", f"⚠ WARNING — above {pv_warn} {pvunit}"
                                else:
                                    sc2, st2 = "ms-pvar-status-ok", "✓ Within normal range"
                                st.markdown(f"""
            <div class="ms-pvar">
              <div class="ms-pvar-label">{pvlabel} ({pvunit})</div>
              <div class="ms-pvar-value">{pv_val:.3g}</div>
              <div class="{sc2}">{st2}</div>
            </div>""", unsafe_allow_html=True)
                            with pvc2:
                                fig_box = px.box(prod_df, x="true_defect_label", y=pvar,
                                                 color="true_defect_label", color_discrete_map=CLASS_COLORS,
                                                 title=f"{pvlabel} distribution by class (synthetic)",
                                                 labels={"true_defect_label": "Class", pvar: f"{pvlabel} ({pvunit})"})
                                fig_box.add_hline(y=pv_val, line_dash="dash", line_color="#e8500a",
                                                  annotation_text=f"This inspection: {pv_val:.3g}",
                                                  annotation_position="top right")
                                fig_box.update_layout(showlegend=False)
                                _chart(fig_box, "pvar_box", 280)

                        with st.expander("All process parameters"):
                            pa, pb, pc, pd_ = st.columns(4)
                            pa.metric("Temperature", f"{proc_row['temperature_c']:.1f} °C")
                            pb.metric("Pressure",    f"{proc_row['pressure_bar']:.2f} bar")
                            pc.metric("Speed",       f"{proc_row['machine_speed_rpm']} rpm")
                            pd_.metric("Vibration",  f"{proc_row['vibration_mms']:.3f} mm/s")
                            pe, pf = st.columns(4)[:2]
                            pe.metric("Humidity",    f"{proc_row['humidity_pct']:.1f} %RH")

                        st.markdown('<div class="ms-section">Station Analysis <span class="tag-synth">Synthetic</span></div>', unsafe_allow_html=True)
                        sa, sb = st.columns(2, gap="large")
                        with sa:
                            sm_sorted = sm_df.sort_values("station_id")
                            fig_ct = go.Figure(go.Bar(
                                x=sm_sorted["station_id"], y=sm_sorted["avg_cycle_time_s"],
                                marker_color=["#dc2626" if s == bn["station_id"] else "#3b82f6" for s in sm_sorted["station_id"]],
                                text=[f"{v:.1f}s" for v in sm_sorted["avg_cycle_time_s"]], textposition="outside",
                            ))
                            fig_ct.update_layout(title="Avg Cycle Time by Station", xaxis_title="Station", yaxis_title="Cycle Time (s)")
                            _chart(fig_ct, "sta_ct", 280)
                            st.caption(f"Station {bn['station_id']} (red) = simulated bottleneck.")
                        with sb:
                            fig_dr = go.Figure(go.Bar(
                                x=sm_sorted["station_id"], y=sm_sorted["defect_rate_pct"],
                                marker_color=["#dc2626" if s == proc_row["station_id"] else "#64748b" for s in sm_sorted["station_id"]],
                                text=[f"{v:.1f}%" for v in sm_sorted["defect_rate_pct"]], textposition="outside",
                            ))
                            fig_dr.update_layout(title=f"Defect Rate by Station (current: {proc_row['station_id']})",
                                                 xaxis_title="Station", yaxis_title="Defect Rate (%)")
                            _chart(fig_dr, "sta_dr", 280)

                        st.markdown('<div class="ms-section">Defect Distribution <span class="tag-synth">Synthetic</span></div>', unsafe_allow_html=True)
                        dc1, dc2 = st.columns(2, gap="large")
                        with dc1:
                            cc = prod_df["true_defect_label"].value_counts().reset_index()
                            cc.columns = ["Class", "Count"]
                            fig_dist = px.bar(cc, x="Class", y="Count", color="Class",
                                              color_discrete_map=CLASS_COLORS, text="Count",
                                              title="All Classes — Volume Distribution")
                            fig_dist.update_layout(showlegend=False)
                            _chart(fig_dist, "dist_all", 260)
                        with dc2:
                            shift_stats = (
                                prod_df.groupby("shift")
                                .apply(lambda g: pd.Series({
                                    "total":   len(g),
                                    "defects": (g["true_defect_label"] != "normal").sum(),
                                }), include_groups=False)
                                .reset_index()
                            )
                            shift_stats["rate"] = (shift_stats["defects"] / shift_stats["total"] * 100).round(1)
                            cur_shift = str(proc_row["shift"])
                            fig_shift = px.bar(
                                shift_stats, x="shift", y="rate", color="shift",
                                color_discrete_map={s: ("#e8500a" if s == cur_shift else "#94a3b8") for s in shift_stats["shift"]},
                                text="rate", title=f"Defect Rate by Shift (current: {cur_shift})",
                                labels={"shift": "Shift", "rate": "Defect Rate (%)"},
                            )
                            fig_shift.update_layout(showlegend=False)
                            fig_shift.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                            _chart(fig_shift, "shift_dr", 260)
                    else:
                        st.info("Synthetic production data not available.")

                with inner[3]:
                    st.warning("**Synthetic economic model** — all monetary values are illustrative only.", icon="⚠️")
                    if econ_summary:
                        cur = econ_summary.get("currency", "USD")
                        unit_cost_map = {
                            "ACCEPT": ("No scrap / rework cost for this unit", 0.0),
                            "REJECT": (f"Estimated scrap cost ({cur})", 25.0),
                            "REVIEW": (f"Estimated rework cost if defective ({cur})", 10.0),
                        }
                        unit_label, unit_cost = unit_cost_map.get(display_dec, ("N/A", 0.0))
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Defect Rate (synthetic)", f"{econ_summary.get('defect_rate_pct', 0):.1f}%")
                        k2.metric(f"Total Scrap ({cur})",    f"{econ_summary.get('total_scrap_cost', 0):,.0f}")
                        k3.metric(f"Total Rework ({cur})",   f"{econ_summary.get('total_rework_cost', 0):,.0f}")
                        k4.metric(f"Downtime Loss ({cur})",  f"{econ_summary.get('total_downtime_loss', 0):,.0f}")
                        st.divider()
                        ec1, ec2 = st.columns(2, gap="large")
                        with ec1:
                            sc2 = econ_summary.get("total_scrap_cost", 0)
                            rw = econ_summary.get("total_rework_cost", 0)
                            dl = econ_summary.get("total_downtime_loss", 0)
                            fig_pie = go.Figure(go.Pie(
                                labels=["Scrap", "Rework", "Downtime"], values=[sc2, rw, dl],
                                hole=0.45, marker_colors=["#dc2626", "#f97316", "#f59e0b"],
                                textinfo="label+percent",
                            ))
                            fig_pie.update_layout(title="Quality Loss Breakdown", legend=dict(orientation="h", y=-0.18))
                            _chart(fig_pie, "pie_cost", 290)
                        with ec2:
                            st.markdown("**Estimated cost for this inspection:**")
                            if display_dec == "ACCEPT":
                                st.success(f"✅ {unit_label}")
                            elif display_dec == "REJECT":
                                st.error(f"❌ {unit_label}: **{cur} {unit_cost:.2f}** (synthetic)")
                            else:
                                st.warning(f"🔍 {unit_label}: **{cur} {unit_cost:.2f}** (if confirmed — synthetic)")
                            total_loss = sc2 + rw + dl
                            total_rev  = econ_summary.get("total_units", 1) * 45.0
                            pct_loss   = (total_loss / total_rev * 100) if total_rev else 0
                            st.metric("Total Quality Loss", f"{cur} {total_loss:,.0f}")
                            st.metric("Loss as % of Revenue", f"{pct_loss:.1f}%")
                            st.metric("Avg Contribution / Unit", f"{cur} {econ_summary.get('avg_contribution_unit', 0):.2f}")
                            st.caption("Assumptions: $45 sell · $25 scrap · $10 rework · $2.50/min downtime — synthetic.")
                    else:
                        st.info("Economic data not available.")

                st.divider()
                if display_dec == "ACCEPT":
                    st.success("✅ **No defect detected.** Continue production and maintain standard monitoring frequency.")
                elif display_dec == "REVIEW":
                    st.warning(
                        f"🔍 **Manual verification required.** Predicted **{predicted_cls.capitalize()}** "
                        f"with {confidence:.1%} confidence — not reliable enough for auto-rejection. "
                        "Route to manual inspection station."
                    )
                else:
                    kb_sum = {
                        "crack":   "Inspect tooling, vibration, and process parameters. Verify with a batch run.",
                        "hole":    "Verify against product drawing first. If defective, inspect tooling and pressure.",
                        "rust":    "Check humidity, coating records, and storage conditions. Quarantine batch if widespread.",
                        "scratch": "Identify the fixed contact point, repair it, and verify with a batch run.",
                    }
                    st.error(
                        f"❌ **{predicted_cls.capitalize()} defect detected** ({confidence:.1%}). "
                        f"{kb_sum.get(predicted_cls, 'Investigate and apply corrective action.')} "
                        "See Root Cause & Corrective Action tab for the full action plan."
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BATCH UPLOAD  (TASK 4)
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('<div class="ms-section">Batch Inspection</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="ms-model-note">
  Upload multiple images at once. ManuSense runs inference on each and
  summarises the results in a table you can download as CSV.
</div>
""", unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload inspection images (multiple)",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="batch_uploader",
    )

    if not batch_files:
        st.markdown(f"""
<div style="text-align:center;padding:40px 32px;color:{TEXT2};
            background:{BG2};border-radius:12px;border:2px dashed {BG3};margin-top:8px;
            animation:fadeInUp 0.5s cubic-bezier(.22,.68,0,1.2) both 0.2s;">
  <div style="font-size:2.2rem;margin-bottom:12px;">📦</div>
  <div style="font-weight:700;color:{TEXT};font-size:1rem;margin-bottom:6px;">
    Select multiple images to batch-inspect
  </div>
  <div style="font-size:0.78rem;color:{TEXT2};margin-top:6px;">
    Hold Ctrl / Cmd to select multiple files in the file picker
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        if not model_ready:
            st.error("Model weights not found.")
        else:
            try:
                predictor_b = _load_predictor()
            except Exception as exc:
                st.error(f"Could not load model: {exc}")
                st.stop()

            batch_results = []
            prog_bar = st.progress(0, text="Running batch inference…")

            for i, bf in enumerate(batch_files):
                try:
                    r = _run_single_inference(bf.getvalue(), bf.name, predictor_b)
                    bp = r["pred"]
                    br = r["rel"]
                    batch_results.append({
                        "File":       bf.name,
                        "Class":      bp.predicted_class.capitalize(),
                        "Confidence": f"{bp.confidence:.1%}",
                        "Decision":   br["display_decision"],
                        "Margin":     f"{br['margin']:.1%}",
                        "Reliable":   "✓" if br["reliable"] else "⚠",
                    })
                    # append to session history
                    b_insp_id = "MS-" + hashlib.md5(bf.name.encode()).hexdigest()[:6].upper()
                    st.session_state.history.append({
                        "insp_id":    b_insp_id,
                        "filename":   bf.name,
                        "timestamp":  datetime.now().strftime("%H:%M:%S"),
                        "class":      bp.predicted_class,
                        "confidence": bp.confidence,
                        "decision":   br["display_decision"],
                        "margin":     br["margin"],
                    })
                    st.session_state.session_counts["total"] += 1
                    st.session_state.session_counts[br["display_decision"]] = (
                        st.session_state.session_counts.get(br["display_decision"], 0) + 1
                    )
                except Exception as exc:
                    batch_results.append({
                        "File": bf.name, "Class": "ERROR",
                        "Confidence": "—", "Decision": "ERROR",
                        "Margin": "—", "Reliable": "✗",
                    })
                prog_bar.progress((i + 1) / len(batch_files),
                                  text=f"Processing {i+1}/{len(batch_files)}: {bf.name}")

            prog_bar.empty()

            df_batch = pd.DataFrame(batch_results)

            # summary KPIs
            total_b  = len(df_batch)
            accept_b = (df_batch["Decision"] == "ACCEPT").sum()
            reject_b = (df_batch["Decision"] == "REJECT").sum()
            review_b = (df_batch["Decision"] == "REVIEW").sum()
            error_b  = (df_batch["Decision"] == "ERROR").sum()

            st.markdown('<div class="ms-section anim-fadeInUp">Batch Summary</div>', unsafe_allow_html=True)
            kb1, kb2, kb3, kb4, kb5 = st.columns(5)
            kb1.metric("Total Images",  total_b)
            kb2.metric("✅ Accept",     accept_b)
            kb3.metric("❌ Reject",     reject_b)
            kb4.metric("🔍 Review",     review_b)
            kb5.metric("Defect/Review Rate", f"{(reject_b+review_b)/total_b*100:.0f}%" if total_b else "—")

            # colour-coded table
            st.markdown('<div class="ms-section">Results Table</div>', unsafe_allow_html=True)

            def _colour_decision(val):
                if val == "ACCEPT": return "color: #16a34a; font-weight:700"
                if val == "REJECT": return "color: #dc2626; font-weight:700"
                if val == "REVIEW": return "color: #d97706; font-weight:700"
                return "color: #94a3b8"

            try:
                styled = df_batch.style.map(_colour_decision, subset=["Decision"])
            except AttributeError:
                styled = df_batch.style.applymap(_colour_decision, subset=["Decision"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # mini bar chart breakdown
            breakdown = pd.DataFrame({
                "Decision": ["ACCEPT", "REJECT", "REVIEW"],
                "Count":    [accept_b, reject_b, review_b],
            })
            fig_batch = px.bar(
                breakdown, x="Decision", y="Count",
                color="Decision",
                color_discrete_map={"ACCEPT": "#16a34a", "REJECT": "#dc2626", "REVIEW": "#d97706"},
                text="Count", title="Batch Decision Breakdown",
            )
            fig_batch.update_layout(showlegend=False)
            _chart(fig_batch, "batch_bar", 260)

            # CSV download
            csv_batch = df_batch.to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Download Batch Report CSV",
                data=csv_batch,
                file_name=f"manusense_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — INSPECTION HISTORY  (TASK 3)
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="ms-section">Inspection History — This Session</div>', unsafe_allow_html=True)

    history = st.session_state.history
    if not history:
        st.markdown(f"""
<div style="text-align:center;padding:40px;color:{TEXT2};background:{BG2};
            border-radius:12px;border:2px dashed {BG3};">
  <div style="font-size:2rem;margin-bottom:10px;">📋</div>
  <div style="font-weight:600;color:{TEXT};">No inspections yet this session.</div>
  <div style="font-size:0.82rem;margin-top:6px;">
    Upload an image in the Inspection tab to begin.
  </div>
</div>
""", unsafe_allow_html=True)
    else:
        sc_h = st.session_state.session_counts
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total", sc_h["total"])
        h2.metric("✅ Accept", sc_h["ACCEPT"])
        h3.metric("❌ Reject", sc_h["REJECT"])
        h4.metric("🔍 Review", sc_h["REVIEW"])

        # trend sparkline
        if len(history) > 1:
            hist_df = pd.DataFrame(history)
            dec_map = {"ACCEPT": 0, "REVIEW": 1, "REJECT": 2}
            hist_df["dec_num"] = hist_df["decision"].map(dec_map)
            fig_hist = px.line(
                hist_df, x=hist_df.index, y="confidence",
                color="decision",
                color_discrete_map={"ACCEPT": "#16a34a", "REJECT": "#dc2626", "REVIEW": "#d97706"},
                markers=True,
                title="Confidence Over Session",
                labels={"index": "Inspection #", "confidence": "Confidence"},
            )
            fig_hist.update_layout(xaxis=dict(tickmode="linear", dtick=1))
            _chart(fig_hist, "hist_spark", 220)

        # rows
        st.markdown('<div class="ms-section">Log</div>', unsafe_allow_html=True)
        dec_icons = {"ACCEPT": "✅", "REJECT": "❌", "REVIEW": "🔍"}
        for i, entry in enumerate(reversed(history)):
            icon    = dec_icons.get(entry["decision"], "⚠️")
            cls_col = CLASS_COLORS.get(entry["class"].lower(), "#6b7280")
            delay   = f"{i * 0.06:.2f}s"
            st.markdown(f"""
<div class="ms-hist-row" style="animation-delay:{delay};">
  <span style="font-weight:700;color:{cls_col};min-width:80px;">{icon} {entry['decision']}</span>
  <span style="font-weight:600;color:{TEXT};min-width:70px;">{entry['class'].capitalize()}</span>
  <span style="color:{TEXT2};min-width:70px;">{entry['confidence']:.1%}</span>
  <span style="color:{TEXT2};min-width:90px;">margin {entry['margin']:.1%}</span>
  <span style="color:{TEXT2};font-size:0.78rem;min-width:100px;">{entry['filename'][:24]}{"…" if len(entry['filename'])>24 else ""}</span>
  <span style="color:{TEXT2};font-size:0.75rem;margin-left:auto;">{entry['timestamp']} · {entry['insp_id']}</span>
</div>
""", unsafe_allow_html=True)

        # export full history
        hist_csv = pd.DataFrame(history).to_csv(index=False).encode()
        st.download_button(
            label="⬇️ Export Full Session History CSV",
            data=hist_csv,
            file_name=f"manusense_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — REFERENCE GALLERY  (TASK 6)
# ══════════════════════════════════════════════════════════════════════════════
with tab_gallery:
    st.markdown('<div class="ms-section">Reference Image Gallery</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="ms-model-note">
  Example Grad-CAM explanations for each class, generated from the training dataset.
  Use these as visual reference when manually verifying a model prediction.
  Warm colours = regions the model focuses on for that class.
</div>
""", unsafe_allow_html=True)

    classes = ["crack", "hole", "normal", "rust", "scratch"]
    cols_gal = st.columns(len(classes), gap="small")

    for i, cls in enumerate(classes):
        with cols_gal[i]:
            path = GRADCAM_PATHS[cls]
            cls_color = CLASS_COLORS[cls]
            st.markdown(f"""
<div class="ms-gal-card" style="animation-delay:{i*0.1:.1f}s;">
""", unsafe_allow_html=True)
            if path.exists():
                img = Image.open(path)
                st.image(img, use_container_width=True)
            else:
                st.markdown(f"""
<div style="height:180px;background:{BG2};border-radius:8px;display:flex;
            align-items:center;justify-content:center;color:{TEXT2};font-size:0.8rem;">
  Image not found
</div>
""", unsafe_allow_html=True)
            st.markdown(f"""
  <div class="ms-gal-label" style="color:{cls_color};">
    {cls.upper()}
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="ms-section">Per-Class Characteristics</div>', unsafe_allow_html=True)

    cls_info = {
        "crack":   ("Linear / branching surface discontinuities.",   "Vibration, stress, tool wear"),
        "hole":    ("Circular / irregular material removal.",         "Tooling wear, pressure drift — verify vs drawing"),
        "normal":  ("No defect — uniform surface texture.",          "N/A"),
        "rust":    ("Discolouration / oxidation patches.",            "Humidity, coating failure, storage time"),
        "scratch": ("Linear surface abrasion marks.",                "Conveyor, handling, fixture contact"),
    }
    ci_cols = st.columns(len(classes), gap="small")
    for i, cls in enumerate(classes):
        desc, causes = cls_info[cls]
        cls_color = CLASS_COLORS[cls]
        with ci_cols[i]:
            st.markdown(f"""
<div style="background:{BG2};border:1px solid {BORDER};border-radius:10px;
            padding:12px 14px;animation:fadeInUp 0.4s {i*0.08:.2f}s both;">
  <div style="font-size:0.78rem;font-weight:800;text-transform:uppercase;
              letter-spacing:.6px;color:{cls_color};margin-bottom:6px;">{cls}</div>
  <div style="font-size:0.82rem;color:{TEXT};margin-bottom:6px;">{desc}</div>
  <div style="font-size:0.75rem;color:{TEXT2};">Root causes: {causes}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — WHAT-IF CLASS EXPLORER  (TASK 7)
# ══════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    st.markdown('<div class="ms-section">What-If Class Explorer</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="ms-model-note">
  Select any defect class and simulate what the model's output would look like.
  Explore the model's probability distribution, confidence behaviour, and the
  action plan for any class — without uploading an image.
</div>
""", unsafe_allow_html=True)

    wi_col1, wi_col2 = st.columns([1, 2], gap="large")

    with wi_col1:
        wi_class = st.selectbox(
            "Select class to explore",
            ["crack", "hole", "normal", "rust", "scratch"],
            format_func=lambda x: x.capitalize(),
            key="wi_class",
        )
        wi_conf = st.slider(
            "Simulated confidence",
            min_value=0.50, max_value=1.00, value=0.95, step=0.01,
            format="%.2f", key="wi_conf",
        )

        # compute simulated decision
        wi_decision = _raw_decision(wi_class, wi_conf)
        wi_dec_icon = {"ACCEPT": "✅", "REJECT": "❌", "REVIEW": "🔍"}[wi_decision]
        wi_card_cls = {"ACCEPT": "ms-accept", "REJECT": "ms-reject"}.get(wi_decision, "ms-review")
        wi_bar_col  = "#16a34a" if wi_decision == "ACCEPT" else "#dc2626" if wi_decision == "REJECT" else "#d97706"

        st.markdown(f"""
<div class="ms-card {wi_card_cls}" style="margin-top:16px;">
  <div class="ms-label">Simulated Decision</div>
  <div class="ms-decision">{wi_dec_icon} {wi_decision}</div>
  <div class="ms-sub">Class: <strong>{wi_class.capitalize()}</strong> ·
   Confidence: <strong>{wi_conf:.1%}</strong> ·
   Threshold: <strong>{st.session_state.conf_threshold:.0%}</strong></div>
</div>
""", unsafe_allow_html=True)

        # confidence bar
        st.markdown(f"""
<div style="margin:12px 0;">
  <div class="ms-conf-wrap">
    <div class="ms-conf-bar" style="width:{int(wi_conf*100)}%;background:{wi_bar_col};"></div>
  </div>
</div>
""", unsafe_allow_html=True)

        # show Grad-CAM reference
        wi_gc_path = GRADCAM_PATHS.get(wi_class)
        if wi_gc_path and wi_gc_path.exists():
            st.markdown('<div class="ms-section">Training Example Grad-CAM</div>', unsafe_allow_html=True)
            st.image(Image.open(wi_gc_path), use_container_width=True)
            st.caption(f"Example Grad-CAM for class '{wi_class}' from the training dataset.")

    with wi_col2:
        # simulate probability distribution
        # put wi_conf on predicted class, spread remainder across others
        other_classes = [c for c in ["crack", "hole", "normal", "rust", "scratch"] if c != wi_class]
        remainder = 1.0 - wi_conf
        # distribute remainder with small random-ish weights seeded by class name
        weights = [ord(c[0]) % 5 + 1 for c in other_classes]
        total_w = sum(weights)
        wi_probs = {c: remainder * w / total_w for c, w in zip(other_classes, weights)}
        wi_probs[wi_class] = wi_conf
        sorted_wi = sorted(wi_probs.items(), key=lambda x: -x[1])

        fig_wi = go.Figure(go.Bar(
            x=[v for _, v in sorted_wi],
            y=[c.capitalize() for c, _ in sorted_wi],
            orientation="h",
            marker_color=[CLASS_COLORS.get(c, "#6b7280") for c, _ in sorted_wi],
            text=[f"{v:.1%}" for _, v in sorted_wi],
            textposition="outside", cliponaxis=False,
        ))
        fig_wi.update_layout(
            xaxis=dict(range=[0, 1.15], tickformat=".0%", title="", showgrid=False),
            yaxis=dict(title=""), title=f"Simulated Probability — {wi_class.capitalize()} @ {wi_conf:.0%}",
        )
        _chart(fig_wi, "wi_probs", 240)

        # Show knowledge base for selected class
        wi_kb = KB[wi_class]
        st.markdown(f'<div class="ms-section">Action Plan for {wi_class.capitalize()}</div>', unsafe_allow_html=True)

        if wi_kb["causes"]:
            with st.expander("Possible Contributing Factors", expanded=True):
                for cause in wi_kb["causes"]:
                    st.markdown(f"• {cause}")

        if wi_decision != "ACCEPT":
            for phase_title, phase_steps in [
                ("🚨 Immediate Actions",     wi_kb["immediate"]),
                ("🔎 Investigation Steps",   wi_kb["investigate"]),
                ("🔧 Corrective Actions",    wi_kb["correct"]),
                ("✅ Verification Steps",    wi_kb["verify"]),
            ]:
                with st.expander(phase_title):
                    for j, s in enumerate(phase_steps, 1):
                        st.markdown(f"**Step {j}:** {s}")
        else:
            st.success("✅ No defect — no corrective action required.")

        if wi_kb["monitor"]:
            with st.expander("📈 Ongoing Monitoring"):
                for m in wi_kb["monitor"]:
                    st.markdown(f"• {m}")

        # threshold impact table
        st.markdown('<div class="ms-section">Threshold Impact</div>', unsafe_allow_html=True)
        st.caption("Shows how the decision changes at different threshold settings for this class & confidence.")
        thr_rows = []
        for thr_val in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            dec = _raw_decision(wi_class, wi_conf, threshold=thr_val)
            thr_rows.append({"Threshold": f"{thr_val:.0%}", "Decision": dec,
                             "Active": "◀ current" if abs(thr_val - st.session_state.conf_threshold) < 0.005 else ""})
        st.dataframe(pd.DataFrame(thr_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown(
        '<div class="ms-section">Held-Out Test Results '
        '<span class="tag-real">Real evaluation · reports/test_metrics.json</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("ResNet18 fine-tuned on challenge dataset (12,000 images · 5 classes). 15% held-out test split.")

    mk = st.columns(4)
    mk[0].metric("Test Accuracy",     "99.11%")
    mk[1].metric("Correct / 1,800",   "1,784")
    mk[2].metric("Incorrect / 1,800", "16")
    mk[3].metric("Weighted F1",       "99.11%")

    st.divider()
    ml, mr = st.columns([1, 1], gap="large")

    with ml:
        st.markdown("**Per-Class Metrics (held-out test set)**")
        st.dataframe(
            pd.DataFrame({
                "Class":     ["crack", "hole", "normal", "rust", "scratch"],
                "Precision": ["1.00",  "0.96", "0.99",   "1.00", "1.00"],
                "Recall":    ["1.00",  "1.00", "1.00",   "0.96", "1.00"],
                "F1-Score":  ["1.00",  "0.98", "1.00",   "0.98", "1.00"],
                "Support":   [360,      360,    360,       360,    360],
            }),
            use_container_width=True, hide_index=True,
        )
        st.markdown("**Dataset Split**")
        st.dataframe(
            pd.DataFrame({
                "Set":      ["Train",        "Validation",    "Test"],
                "Images":   [8400,            1800,            1800],
                "Strategy": ["70% stratified","15% stratified","15% stratified"],
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption("Stratified split · random seed 42 · no leakage between splits")

    with mr:
        st.markdown("**Confusion Matrix**")
        cm = np.array([
            [360,  0,  0,  0,  0],
            [  0,360,  0,  0,  0],
            [  0,  0,360,  0,  0],
            [  0, 14,  2,344,  0],
            [  0,  0,  0,  0,360],
        ])
        classes_cm = ["crack", "hole", "normal", "rust", "scratch"]
        fig_cm = px.imshow(
            cm, x=classes_cm, y=classes_cm,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Predicted", y="True", color="Count"),
            title="Confusion Matrix — Test Set",
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        _chart(fig_cm, "conf_matrix", 360)
        st.caption("Primary failure mode: rust → hole (14 errors). All other pairs: 0 errors.")

    st.divider()
    st.markdown("**Data Provenance**")
    c_real, c_synth = st.columns(2)
    with c_real:
        st.success("""
**Real data used in this prototype**
- 12,000 challenge inspection images
- 5 classes: crack · hole · normal · rust · scratch
- Trained ResNet18 checkpoint (best_model.pt)
- Model predictions & probabilities from checkpoint
- Grad-CAM from the trained model
- Test metrics from held-out evaluation
        """)
    with c_synth:
        st.warning("""
**Synthetic demonstration data (clearly labelled)**
- 500 production / process records — not provided with challenge
- Station cycle times, downtime, WIP — simulated
- Bottleneck analysis — synthetic
- Economics (scrap, rework, downtime cost) — illustrative only
        """)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ms-footer">
  <strong>ManuSense v2.0</strong> &nbsp;·&nbsp;
  AI-Powered Manufacturing Quality &amp; Root-Cause Intelligence &nbsp;·&nbsp;
  "ManuSense doesn't just detect defects — it understands the manufacturing process behind them."
  <br>
  ResNet18 · 99.11% held-out accuracy · PyTorch · Streamlit · Advisory only · No machine control
</div>
""", unsafe_allow_html=True)
