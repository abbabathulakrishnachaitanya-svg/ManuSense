"""
app/main.py
────────────
ManuSense — AI-Powered Manufacturing Quality & Root-Cause Intelligence
Streamlit dashboard entry point.

Run with (from the project root):
    streamlit run app/main.py

This file handles ONLY the UI layer.
All ML logic lives in app/vision/predictor.py and app/vision/preprocessing.py.
"""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from pathlib import Path

import streamlit as st

# ── Make sure project root is on sys.path when Streamlit launches the script ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings

logging.basicConfig(level=settings.log_level)
log = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ManuSense — Visual Inspection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Decision colour map ───────────────────────────────────────────────────────
_DECISION_STYLE = {
    "ACCEPT": ("✅ ACCEPT",  "#1a7a1a", "#d4edda"),
    "REJECT": ("❌ REJECT",  "#8b0000", "#f8d7da"),
    "REVIEW": ("🔍 REVIEW",  "#7a5200", "#fff3cd"),
}

# ── Cached model loader ───────────────────────────────────────────────────────
# st.cache_resource keeps the model in memory across Streamlit reruns so we
# don't reload it on every interaction.

@st.cache_resource(show_spinner="Loading model …")
def _load_predictor():
    """Load the DefectPredictor once and cache it for the session."""
    from app.vision.predictor import DefectPredictor

    predictor = DefectPredictor()
    predictor.load_model()
    return predictor


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏭 ManuSense")
    st.caption("AI-Powered Manufacturing Quality & Root-Cause Intelligence")
    st.divider()

    st.subheader("Navigation")
    page = st.radio(
        label="Go to",
        options=["Visual Inspection", "About"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Model configuration")
    st.caption(f"Architecture: `{settings.model_name}`")
    st.caption(f"Weights: `{settings.model_weights_path}`")
    st.caption(f"Confidence threshold: `{settings.confidence_threshold}`")
    st.caption(f"Uncertainty threshold: `{settings.uncertainty_threshold}`")
    st.info(
        "Thresholds are configurable via `.env`.  "
        "Tune them after reviewing the confusion matrix.",
        icon="ℹ️",
    )


# ── Page: Visual Inspection ───────────────────────────────────────────────────

def page_visual_inspection() -> None:
    st.title("Visual Inspection")
    st.markdown(
        "Upload a surface inspection image.  "
        "The model will classify it and issue an **ACCEPT / REJECT / REVIEW** decision."
    )

    # ── Check model weights exist ─────────────────────────────────────────────
    if not settings.model_weights_path.exists():
        st.error(
            f"No trained model found at `{settings.model_weights_path}`.\n\n"
            "**Train the model first:**\n"
            "```\npython -m app.vision.train\n```",
            icon="🚫",
        )
        return

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        predictor = _load_predictor()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}", icon="🚫")
        log.exception("Model load error")
        return

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload an inspection image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Grayscale or colour images accepted (256×256 recommended).",
    )

    if uploaded is None:
        st.info("Upload an image above to start inspection.", icon="📁")
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    with st.spinner("Analysing image …"):
        try:
            # Save the upload to a temp file so predictor can open it with PIL
            import tempfile, os
            suffix = Path(uploaded.name).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            result = predictor.predict(tmp_path)
        except Exception as exc:
            st.error(f"Inference failed: {exc}", icon="🚫")
            log.exception("Inference error")
            return
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Layout ────────────────────────────────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded image")
        st.image(uploaded, use_column_width=True, caption=uploaded.name)

    with col_result:
        st.subheader("Inspection result")

        # Decision badge
        label, text_color, bg_color = _DECISION_STYLE.get(
            result.decision, ("⚠️ UNKNOWN", "#555", "#eee")
        )
        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                border-left: 6px solid {text_color};
                padding: 16px 20px;
                border-radius: 6px;
                margin-bottom: 12px;
            ">
                <span style="font-size:1.6rem; font-weight:700; color:{text_color};">
                    {label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Key metrics
        m1, m2 = st.columns(2)
        m1.metric("Predicted class", result.predicted_class.capitalize())
        m2.metric("Confidence", f"{result.confidence:.1%}")

        if result.is_uncertain:
            st.warning(
                "Confidence is below the uncertainty threshold — "
                "manual review recommended.",
                icon="⚠️",
            )

        st.divider()

        # Probability bar chart
        st.subheader("Class probabilities")
        _render_probability_bars(result.probabilities)

        st.divider()
        st.caption(
            "All decisions are advisory only. "
            "No machine or process control is performed."
        )


def _render_probability_bars(probabilities: dict[str, float]) -> None:
    """Render a styled horizontal bar chart for class probabilities."""
    import plotly.graph_objects as go

    classes = list(probabilities.keys())
    values  = [probabilities[c] for c in classes]

    # Colour bars: green for normal, red for defects
    colors = [
        "#2e8b57" if c == "normal" else "#c0392b"
        for c in classes
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=[c.capitalize() for c in classes],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        yaxis=dict(title=""),
        height=280,
        margin=dict(l=10, r=60, t=10, b=30),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Page: About ───────────────────────────────────────────────────────────────

def page_about() -> None:
    st.title("About ManuSense")
    st.markdown(
        """
        **ManuSense** is a software-only AI assistant for manufacturing quality teams.

        ### Visual Inspection module
        | Item | Detail |
        |------|--------|
        | Architecture | ResNet18 (fine-tuned, pretrained ImageNet weights) |
        | Classes | crack · hole · normal · rust · scratch |
        | Input | 256×256 grayscale PNG → 224×224 3-channel tensor |
        | Training data | 12 000 images (2 400 per class) |
        | Decision states | ACCEPT · REJECT · REVIEW |

        ### Constraints
        - **No live camera** — image upload only
        - **No machine control** — all decisions are advisory
        - **No external API** — runs fully locally

        ### Roadmap (post-hackathon)
        - Root-cause correlation with process parameters
        - Production bottleneck and throughput analysis
        - Profitability simulation (scrap/rework costs)
        - Grad-CAM defect localisation overlay
        """
    )


# ── Router ────────────────────────────────────────────────────────────────────
if page == "Visual Inspection":
    page_visual_inspection()
elif page == "About":
    page_about()
