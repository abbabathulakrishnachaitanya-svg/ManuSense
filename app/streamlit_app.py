"""
app/streamlit_app.py — ManuSense v3.0
AI-Powered Manufacturing Quality & Root-Cause Intelligence

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="ManuSense",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.components.styles import inject_css, COLORS
from app.components.badges import decision_badge

inject_css()

# ── Session state defaults ────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "session_counts" not in st.session_state:
    st.session_state.session_counts = {"total": 0, "ACCEPT": 0, "REJECT": 0, "REVIEW": 0}
if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = 0.80
if "current_page" not in st.session_state:
    st.session_state.current_page = "Inspection"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown(f"""
<div style="padding:8px 0 16px 0;">
  <div style="font-size:1.35rem;font-weight:900;letter-spacing:-0.3px;color:{COLORS['text']};">
    🏭 Manu<span style="color:{COLORS['accent']};">Sense</span>
  </div>
  <div style="font-size:0.72rem;color:{COLORS['text2']};margin-top:2px;font-style:italic;">
    AI Manufacturing Quality Intelligence
  </div>
</div>
""", unsafe_allow_html=True)

    # Navigation
    pages = [
        "🔍  Inspection",
        "📊  Quality Dashboard",
        "🔎  Root-Cause Investigation",
        "🗂️  Data Explorer",
        "📐  Model Validation",
        "ℹ️  About",
    ]
    page_labels = {p.split("  ")[1]: p for p in pages}
    page_keys   = [p.split("  ")[1] for p in pages]

    sel = st.radio("Navigation", pages, label_visibility="collapsed",
                   key="nav_radio")
    current = sel.split("  ")[1].strip()

    st.divider()

    # Confidence threshold
    st.markdown(f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["text"]};">⚙️ Decision Threshold</div>',
                unsafe_allow_html=True)
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.50, max_value=0.99,
        value=st.session_state.conf_threshold,
        step=0.01, format="%.2f",
        label_visibility="collapsed",
        help="Predictions ≥ this value → ACCEPT/REJECT. Below → REVIEW.",
        key="threshold_slider",
    )
    st.session_state.conf_threshold = threshold
    st.caption(f"Auto-decide at **{threshold:.0%}** · below = Manual Review")

    st.divider()

    # Session summary
    sc = st.session_state.session_counts
    st.markdown(f'<div style="font-size:0.8rem;font-weight:700;color:{COLORS["text"]};">📋 Session</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:5px;margin:8px 0;">
  <div style="display:flex;justify-content:space-between;padding:6px 10px;
              background:rgba(35,134,54,0.1);border-radius:6px;border-left:3px solid {COLORS['accept']};">
    <span style="font-size:0.8rem;color:#3fb950;">✅ Accept</span>
    <span style="font-size:0.9rem;font-weight:800;color:{COLORS['text']};">{sc['ACCEPT']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 10px;
              background:rgba(218,54,51,0.1);border-radius:6px;border-left:3px solid {COLORS['reject']};">
    <span style="font-size:0.8rem;color:#f85149;">❌ Reject</span>
    <span style="font-size:0.9rem;font-weight:800;color:{COLORS['text']};">{sc['REJECT']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 10px;
              background:rgba(210,153,34,0.1);border-radius:6px;border-left:3px solid {COLORS['review']};">
    <span style="font-size:0.8rem;color:#e3b341;">🔍 Review</span>
    <span style="font-size:0.9rem;font-weight:800;color:{COLORS['text']};">{sc['REVIEW']}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 10px;
              background:{COLORS['bg3']};border-radius:6px;">
    <span style="font-size:0.8rem;color:{COLORS['text2']};">Total</span>
    <span style="font-size:0.9rem;font-weight:800;color:{COLORS['text']};">{sc['total']}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    if sc["total"] > 0:
        defect_rate = (sc["REJECT"] + sc["REVIEW"]) / sc["total"] * 100
        st.progress(min(int(defect_rate), 100))
        st.caption(f"Defect/Review rate: **{defect_rate:.0f}%**")
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.history = []
            st.session_state.session_counts = {"total": 0, "ACCEPT": 0, "REJECT": 0, "REVIEW": 0}
            st.rerun()

    st.divider()
    st.caption("ManuSense v3.0 · ResNet18 · 99.11% test accuracy · Advisory only")

# ── Top bar ───────────────────────────────────────────────────────────────────
from app.config import settings as _app_settings
model_ready = _app_settings.model_weights_path.exists() or (PROJECT_ROOT / "models" / "best_model.pt").exists()
pill = (
    f'<span style="font-size:0.72rem;font-weight:700;color:#3fb950;'
    f'background:rgba(35,134,54,0.15);border:1px solid {COLORS["accept"]};'
    f'border-radius:99px;padding:4px 12px;">● Model Ready · ResNet18 · 99.11%</span>'
    if model_ready else
    f'<span style="font-size:0.72rem;font-weight:700;color:#e3b341;'
    f'background:rgba(210,153,34,0.15);border:1px solid {COLORS["review"]};'
    f'border-radius:99px;padding:4px 12px;">● Model not found</span>'
)

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:12px 0 14px 0;border-bottom:1px solid {COLORS['border']};
            margin-bottom:24px;">
  <div style="display:flex;align-items:center;gap:12px;">
    <svg width="34" height="34" viewBox="0 0 52 52" fill="none">
      <rect width="52" height="52" rx="10" fill="{COLORS['accent']}"/>
      <rect x="8" y="30" width="6" height="14" fill="white"/>
      <rect x="16" y="24" width="6" height="20" fill="white"/>
      <rect x="24" y="18" width="6" height="26" fill="white"/>
      <rect x="32" y="22" width="6" height="22" fill="white"/>
      <rect x="40" y="28" width="4" height="16" fill="white"/>
      <circle cx="11" cy="26" r="2.5" fill="#fbbf24"/>
      <circle cx="19" cy="20" r="2.5" fill="#fbbf24"/>
      <circle cx="27" cy="14" r="2.5" fill="#fbbf24"/>
      <circle cx="35" cy="18" r="2.5" fill="#fbbf24"/>
      <polyline points="11,26 19,20 27,14 35,18" stroke="#fbbf24" stroke-width="1.5" fill="none"/>
    </svg>
    <div>
      <div style="font-size:1.45rem;font-weight:900;color:{COLORS['text']};letter-spacing:-0.4px;">
        Manu<span style="color:{COLORS['accent']};">Sense</span>
      </div>
      <div style="font-size:0.72rem;color:{COLORS['text2']};font-style:italic;">
        AI-Powered Manufacturing Quality &amp; Root-Cause Intelligence
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    {pill}
    <span style="font-size:0.7rem;color:{COLORS['text2']};background:{COLORS['bg2']};
                 border:1px solid {COLORS['border']};border-radius:99px;padding:4px 10px;">
      Threshold: {threshold:.0%}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Route to page ─────────────────────────────────────────────────────────────
from app.pages import inspection, quality_dashboard, root_cause, data_explorer, model_validation, about

if current == "Inspection":
    inspection.render()
elif current == "Quality Dashboard":
    quality_dashboard.render()
elif current == "Root-Cause Investigation":
    root_cause.render()
elif current == "Data Explorer":
    data_explorer.render()
elif current == "Model Validation":
    model_validation.render()
elif current == "About":
    about.render()
