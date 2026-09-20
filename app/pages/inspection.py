"""
app/pages/inspection.py
────────────────────────
Page 1 — AI Visual Inspection

Workflow: Upload → ResNet18 inference → Confidence/Decision →
          Grad-CAM explanation → QC dataset context bridge →
          Possible contributing factors → Corrective-action guidance
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.badges import (
    decision_badge, disclaimer, finding_block,
    real_badge, section_heading, synth_badge,
)
from app.components.charts import confidence_gauge, prob_bar
from app.components.styles import COLORS
from app.data.qc_data import (
    VISION_TO_QC,
    association_analysis,
    corrective_action_guidance,
    defect_type_by_line,
    defects_by_type,
    load_normalised,
)

log = logging.getLogger(__name__)

# ── Vision-model class → display label ───────────────────────────────────────
CLASS_DISPLAY = {
    "normal":  "Normal — No Defect",
    "crack":   "Crack",
    "hole":    "Hole",
    "rust":    "Rust / Corrosion",
    "scratch": "Surface Scratch",
}


@st.cache_resource(show_spinner="Loading ResNet18 model…")
def _load_predictor():
    from app.vision.predictor import DefectPredictor
    p = DefectPredictor()
    p.load_model()
    return p


def render() -> None:
    # ── page title
    st.markdown(
        section_heading("🔍 AI Visual Inspection", real_badge()),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:{COLORS["text2"]};font-size:0.88rem;margin-top:-8px;margin-bottom:20px;">'
        "Upload a manufacturing / product surface image. The ResNet18 model classifies the defect, "
        "provides a confidence-aware decision, and connects the result to historical QC context.</p>",
        unsafe_allow_html=True,
    )

    # ── model ready check — try settings path first, fall back to PROJECT_ROOT
    from app.config import settings as _s
    model_path = _s.model_weights_path if _s.model_weights_path.exists() \
                 else PROJECT_ROOT / "models" / "best_model.pt"
    if not model_path.exists():
        st.error("**Model weights not found.** Run `python -m app.vision.train` first.")
        return

    # ── threshold control (sidebar or inline)
    threshold = st.session_state.get("conf_threshold", 0.80)

    # ── upload
    uploaded = st.file_uploader(
        "Upload inspection image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed",
        key="insp_upload",
    )

    if uploaded is None:
        st.markdown(f"""
<div style="text-align:center;padding:48px 32px;background:{COLORS['bg2']};
            border-radius:12px;border:2px dashed {COLORS['border']};margin-top:8px;">
  <div style="font-size:2.4rem;margin-bottom:12px;">📷</div>
  <div style="font-weight:700;color:{COLORS['text']};font-size:1rem;margin-bottom:6px;">
    Drop a surface inspection image here
  </div>
  <div style="color:{COLORS['text2']};font-size:0.84rem;">
    PNG · JPG · BMP · TIFF &nbsp;|&nbsp; Best results: grayscale surface texture images
  </div>
</div>
""", unsafe_allow_html=True)
        return

    # ── load model + QC data
    try:
        predictor = _load_predictor()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    try:
        qc_df = load_normalised()
    except Exception as exc:
        st.warning(f"QC dataset unavailable: {exc}")
        qc_df = None

    # ── run inference
    scan_ph = st.empty()
    scan_ph.markdown(f"""
<div style="padding:14px 18px;background:{COLORS['bg2']};border-radius:10px;
            border:1px solid {COLORS['border']};margin-bottom:12px;">
  <div style="font-size:0.88rem;font-weight:600;color:{COLORS['text']};margin-bottom:6px;">
    🔬 Running ResNet18 inference…
  </div>
  <div style="height:6px;background:{COLORS['border']};border-radius:99px;overflow:hidden;">
    <div style="height:6px;width:100%;background:linear-gradient(90deg,transparent,{COLORS['accent']},transparent);
                border-radius:99px;animation:scanBar 1.2s linear infinite;"></div>
  </div>
</div>
<style>@keyframes scanBar{{0%{{margin-left:-40%}}100%{{margin-left:100%}}}}</style>
""", unsafe_allow_html=True)

    pred = tensor = None
    try:
        suffix = Path(uploaded.name).suffix or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getvalue())
            tmp = f.name
        pred = predictor.predict(tmp)
        from app.vision.preprocessing import preprocess_single_image
        tensor = preprocess_single_image(tmp)
    except Exception as exc:
        scan_ph.empty()
        st.error(f"Inference failed: {exc}")
        return
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

    scan_ph.empty()

    # ── DOMAIN SHIFT DETECTION ────────────────────────────────────────────────
    # The model was trained on 256×256 greyscale industrial surface textures.
    # Real-world photos are outside that distribution. We detect them using
    # raw PIL image properties BEFORE the pipeline strips colour information.

    def _detect_domain_shift(uploaded_bytes: bytes) -> dict:
        from PIL import Image as PILImage
        import io, numpy as np
        reasons = []
        try:
            pil_img = PILImage.open(io.BytesIO(uploaded_bytes))
            w, h = pil_img.size
            mode = pil_img.mode

            # Signal 1: large resolution → real photo, not a surface scan
            if max(w, h) >= 600:
                reasons.append(
                    f"Resolution {w}×{h} px — training images are 256×256 px surface textures."
                )

            # Signal 2: colour image with significant inter-channel difference
            if mode in ("RGB", "RGBA"):
                arr = np.array(pil_img.convert("RGB")).astype(float)
                ch_means = arr.mean(axis=(0, 1))   # R, G, B means (0–255 scale)
                ch_spread = float(np.std(ch_means))
                if ch_spread > 15:
                    reasons.append(
                        f"Colour photograph detected (channel spread {ch_spread:.0f}/255). "
                        "The model was trained on greyscale surface textures."
                    )

            # Signal 3: non-square aspect ratio > 1.6
            ratio = max(w, h) / max(min(w, h), 1)
            if ratio > 1.6:
                reasons.append(
                    f"Aspect ratio {ratio:.1f}:1 — surface scans are typically square."
                )

        except Exception:
            pass

        is_ood = len(reasons) >= 2
        severity = "high" if len(reasons) >= 3 else "medium" if is_ood else "low"
        return {"is_ood": is_ood, "reasons": reasons, "severity": severity}

    ood = _detect_domain_shift(uploaded.getvalue())

    if ood["is_ood"]:
        scan_ph.empty()
        st.markdown(f"""
<div style="background:#450a0a;border:2px solid {COLORS['reject']};border-radius:12px;
            padding:24px 28px;margin-bottom:20px;">
  <div style="font-size:1.4rem;font-weight:900;color:#f85149;margin-bottom:10px;">
    ⚠️ INVALID IMAGE — Outside Model Domain
  </div>
  <div style="font-size:0.9rem;color:{COLORS['text']};margin-bottom:14px;line-height:1.7;">
    This image does not appear to be a manufacturing surface inspection image.
    The ResNet18 model was trained on <strong>greyscale industrial surface textures (256×256 px)</strong>
    and cannot reliably classify images from a different domain.<br><br>
    <strong>Softmax always outputs a confident-looking result even for irrelevant images —
    showing that result here would be misleading.</strong>
  </div>
  <div style="font-size:0.82rem;color:{COLORS['text2']};margin-bottom:8px;font-weight:700;
              text-transform:uppercase;letter-spacing:.5px;">
    Detection signals ({len(ood['reasons'])}):
  </div>
  <ul style="margin:0;padding-left:18px;">
    {''.join(f'<li style="font-size:0.84rem;color:{COLORS["text2"]};margin-bottom:5px;">{r}</li>' for r in ood["reasons"])}
  </ul>
  <div style="margin-top:16px;padding-top:14px;border-top:1px solid rgba(248,81,73,0.3);
              font-size:0.78rem;color:{COLORS['text2']};font-style:italic;">
    Please upload a greyscale surface texture image — e.g. from
    <code>reports/gradcam/</code> or any industrial surface scan (≤256×256 px).
  </div>
</div>
""", unsafe_allow_html=True)
        return
    def _decision(cls: str, conf: float) -> str:
        low = 0.50
        if conf < low:
            return "REVIEW"
        if cls == "normal":
            return "ACCEPT" if conf >= threshold else "REVIEW"
        return "REJECT" if conf >= threshold else "REVIEW"

    raw_decision = _decision(pred.predicted_class, pred.confidence)
    pred_cls = pred.predicted_class
    confidence = pred.confidence

    # Soft OOD warning (1 signal = borderline, still show result but warn)
    soft_ood = len(ood["reasons"]) == 1

    # ── Grad-CAM
    gc_overlay = gc_error = None
    try:
        from app.vision.explainability import explain_prediction
        gc_result = explain_prediction(model=predictor._model, tensor=tensor)
        gc_overlay = gc_result["overlay"]
    except Exception as exc:
        gc_error = str(exc)

    # ── append to session history
    insp_id = "MS-" + hashlib.md5(uploaded.name.encode()).hexdigest()[:6].upper()
    hist = st.session_state.setdefault("history", [])
    counts = st.session_state.setdefault(
        "session_counts", {"total": 0, "ACCEPT": 0, "REJECT": 0, "REVIEW": 0}
    )
    if not hist or hist[-1]["insp_id"] != insp_id:
        hist.append({
            "insp_id":    insp_id,
            "filename":   uploaded.name,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "class":      pred_cls,
            "confidence": confidence,
            "decision":   raw_decision,
            "margin":     max(pred.probabilities.values()) -
                          sorted(pred.probabilities.values())[-2],
        })
        counts["total"] += 1
        counts[raw_decision] = counts.get(raw_decision, 0) + 1

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — Image + Decision
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    col_img, col_dec = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown(section_heading("Uploaded Image"), unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)

    with col_dec:
        st.markdown(section_heading("Inspection Result"), unsafe_allow_html=True)

        # ── AI Prediction (always shown, never suppressed)
        st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-radius:10px;padding:18px 20px;margin-bottom:12px;">
  <div style="font-size:0.7rem;font-weight:800;text-transform:uppercase;
              letter-spacing:.7px;color:{COLORS['text2']};margin-bottom:4px;">
    AI Model Prediction
  </div>
  <div style="font-size:1.7rem;font-weight:900;color:{COLORS['text']};">
    {CLASS_DISPLAY.get(pred_cls, pred_cls.capitalize())}
  </div>
  <div style="font-size:0.82rem;color:{COLORS['text2']};margin-top:4px;">
    Confidence: <strong style="color:{COLORS['text']};">{confidence:.1%}</strong>
    &nbsp;·&nbsp; Inspection ID: <strong style="color:{COLORS['accent']};">{insp_id}</strong>
  </div>
</div>
""", unsafe_allow_html=True)

        # ── Operational Decision (separate from prediction)
        dec_colors = {
            "ACCEPT": (COLORS["accept"], "#052e16"),
            "REJECT": (COLORS["reject"], "#450a0a"),
            "REVIEW": (COLORS["review"], "#422006"),
        }
        border_c, bg_c = dec_colors.get(raw_decision, (COLORS["review"], COLORS["bg2"]))

        if raw_decision == "ACCEPT":
            dec_text = "No defect detected at current confidence threshold. Part passes triage."
        elif raw_decision == "REJECT":
            dec_text = (
                f"<strong>{pred_cls.capitalize()}</strong> defect detected with "
                f"{confidence:.1%} confidence. Flag for removal from production flow."
            )
        else:
            dec_text = (
                f"Confidence ({confidence:.1%}) is below the prototype review threshold "
                f"({threshold:.0%}). Human inspection required before any disposition."
            )

        st.markdown(f"""
<div style="background:{bg_c};border:2px solid {border_c};border-radius:10px;
            padding:18px 20px;margin-bottom:12px;">
  <div style="font-size:0.7rem;font-weight:800;text-transform:uppercase;
              letter-spacing:.7px;color:{COLORS['text2']};margin-bottom:4px;">
    Operational Decision
  </div>
  <div style="font-size:2rem;font-weight:900;color:{border_c};margin-bottom:6px;">
    {decision_badge(raw_decision)}
  </div>
  <div style="font-size:0.84rem;color:{COLORS['text']};">{dec_text}</div>
  <div style="font-size:0.73rem;color:{COLORS['text2']};margin-top:8px;font-style:italic;">
    Prototype review threshold: {threshold:.0%} — configurable in sidebar.
    Advisory only — no machine control is performed.
  </div>
</div>
""", unsafe_allow_html=True)

        # ── KPI strip
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Class", pred_cls.capitalize())
        m2.metric("Confidence", f"{confidence:.1%}")
        sorted_p = sorted(pred.probabilities.values(), reverse=True)
        margin = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else 0
        m3.metric("Prediction Margin", f"{margin:.1%}",
                  help="Gap between top-1 and top-2 probabilities. <30% = low margin.")

        # soft OOD warning
        if soft_ood:
            st.markdown(f"""
<div style="background:rgba(210,153,34,0.1);border:1px solid {COLORS['review']};
            border-left:4px solid {COLORS['review']};border-radius:0 8px 8px 0;
            padding:10px 14px;margin-top:8px;font-size:0.83rem;color:{COLORS['text2']};">
  ⚠️ <strong style="color:{COLORS['review']};">Domain caution:</strong>
  {ood['reasons'][0]}
  This result may be unreliable. Use with caution.
</div>
""", unsafe_allow_html=True)

        # ── confidence gauge
        confidence_gauge(confidence, raw_decision, key="insp_gauge")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — Inner tabs
    # ══════════════════════════════════════════════════════════════════════════
    inner = st.tabs([
        "📊 Probability Breakdown",
        "🔬 Model Attention (Grad-CAM)",
        "📋 QC Historical Context",
        "🛠️ Contributing Factors & Actions",
    ])

    # ── Tab B1: Probabilities
    with inner[0]:
        prob_bar(pred.probabilities, pred_cls, key="insp_prob")
        st.markdown(disclaimer(
            "Class probabilities show the model's output distribution. "
            "High probability does not guarantee correct classification on out-of-distribution images."
        ), unsafe_allow_html=True)

    # ── Tab B2: Grad-CAM
    with inner[1]:
        if gc_overlay is not None:
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown(section_heading("Original Image"), unsafe_allow_html=True)
                st.image(uploaded, use_container_width=True)
            with c2:
                st.markdown(
                    section_heading("Model Attention Visualization", real_badge()),
                    unsafe_allow_html=True,
                )
                st.image(gc_overlay, use_container_width=True)

            st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-left:4px solid {COLORS['blue']};border-radius:0 8px 8px 0;
            padding:14px 18px;margin-top:12px;font-size:0.85rem;color:{COLORS['text2']};">
  <strong style="color:{COLORS['text']};">How to read this:</strong>
  Warm colours (red/yellow) highlight image regions that contributed strongly
  to the model's prediction. Cool colours had low influence.<br><br>
  <strong style="color:{COLORS['text']};">Important:</strong>
  This is a <em>model explanation aid</em>, not ground-truth defect localisation.
  The dataset contains no bounding-box annotations.
  Grad-CAM shows where the model focused — not a verified defect region.
</div>
""", unsafe_allow_html=True)
        else:
            st.warning(
                f"Grad-CAM could not be generated: `{gc_error}`\n\n"
                "The classification result above is still valid."
            )

    # ── Tab B3: QC Historical Context
    with inner[2]:
        st.markdown(
            section_heading("Historical QC Context", real_badge()),
            unsafe_allow_html=True,
        )
        st.markdown(f"""
<p style="color:{COLORS['text2']};font-size:0.84rem;margin-bottom:16px;">
The charts below are drawn from the <strong>Infoveave Product Quality Control sample dataset
(944 inspection records, 2022–2024)</strong>. They show historical patterns associated with
defect types that are visually related to the model's prediction.<br>
<em>These are observed associations — not proven causal relationships.</em>
</p>
""", unsafe_allow_html=True)

        if qc_df is None:
            st.info("QC dataset not available.")
        else:
            related_qc = VISION_TO_QC.get(pred_cls, [])

            if pred_cls == "normal" or not related_qc:
                st.success(
                    "✅ Model predicted **Normal** — no defect detected. "
                    "No QC defect context is applicable for this prediction."
                )
            else:
                st.markdown(f"""
<div style="background:{COLORS['bg3']};border:1px solid {COLORS['border']};
            border-radius:8px;padding:12px 16px;margin-bottom:16px;font-size:0.84rem;">
  <strong style="color:{COLORS['accent']};">Visual prediction:</strong>
  <strong style="color:{COLORS['text']};">{pred_cls.capitalize()}</strong>
  &nbsp;→&nbsp;
  <strong style="color:{COLORS['accent']};">Related QC defect types:</strong>
  <strong style="color:{COLORS['text']};">{', '.join(related_qc)}</strong>
</div>
""", unsafe_allow_html=True)

                # filter QC data to related types
                related_df = qc_df[qc_df["defect_type"].isin(related_qc)]
                all_defect_df = qc_df.dropna(subset=["defect_type"])

                c_left, c_right = st.columns(2, gap="large")

                with c_left:
                    from app.components.charts import defect_bar, line_defect_bar
                    type_counts = defects_by_type(qc_df)
                    # highlight related types
                    import plotly.graph_objects as go
                    from app.components.styles import DEFECT_COLORS
                    colors = [
                        COLORS["accent"] if t in related_qc else COLORS["border"]
                        for t in type_counts["defect_type"]
                    ]
                    fig = go.Figure(go.Bar(
                        x=type_counts["count"],
                        y=type_counts["defect_type"],
                        orientation="h",
                        marker_color=colors,
                        text=type_counts["count"],
                        textposition="outside",
                        cliponaxis=False,
                    ))
                    fig.update_layout(
                        title="All Defect Types — related highlighted",
                        height=280,
                        paper_bgcolor=COLORS["bg2"],
                        plot_bgcolor=COLORS["bg2"],
                        font=dict(color=COLORS["text"]),
                        margin=dict(l=16, r=16, t=36, b=12),
                        xaxis=dict(title="Count", gridcolor=COLORS["border"]),
                        yaxis=dict(gridcolor=COLORS["border"]),
                    )
                    st.plotly_chart(fig, use_container_width=True, key="ctx_all_types")

                with c_right:
                    if not related_df.empty:
                        line_counts = related_df["line_id"].value_counts().reset_index()
                        line_counts.columns = ["line_id", "count"]
                        from app.components.styles import LINE_COLORS
                        lcolors = [LINE_COLORS.get(l, COLORS["accent"]) for l in line_counts["line_id"]]
                        fig2 = go.Figure(go.Bar(
                            x=line_counts["line_id"],
                            y=line_counts["count"],
                            marker_color=lcolors,
                            text=line_counts["count"],
                            textposition="outside",
                        ))
                        fig2.update_layout(
                            title=f"{', '.join(related_qc)} by Production Line",
                            height=280,
                            paper_bgcolor=COLORS["bg2"],
                            plot_bgcolor=COLORS["bg2"],
                            font=dict(color=COLORS["text"]),
                            margin=dict(l=16, r=16, t=36, b=12),
                            xaxis=dict(title="Line", gridcolor=COLORS["border"]),
                            yaxis=dict(title="Count", gridcolor=COLORS["border"]),
                        )
                        st.plotly_chart(fig2, use_container_width=True, key="ctx_line")
                    else:
                        st.info(f"No records for {', '.join(related_qc)} in the dataset.")

                # monthly trend for related types
                if not related_df.empty:
                    monthly = related_df.groupby("year_month").size().reset_index(name="count")
                    monthly = monthly.sort_values("year_month")
                    import plotly.graph_objects as go
                    fig3 = go.Figure(go.Scatter(
                        x=monthly["year_month"], y=monthly["count"],
                        mode="lines+markers",
                        line=dict(color=COLORS["accent"], width=2),
                        marker=dict(size=5),
                    ))
                    fig3.update_layout(
                        title=f"Monthly Trend — {', '.join(related_qc)}",
                        height=240,
                        paper_bgcolor=COLORS["bg2"],
                        plot_bgcolor=COLORS["bg2"],
                        font=dict(color=COLORS["text"]),
                        margin=dict(l=16, r=16, t=36, b=12),
                        xaxis=dict(title="Month", gridcolor=COLORS["border"]),
                        yaxis=dict(title="Records", gridcolor=COLORS["border"]),
                    )
                    st.plotly_chart(fig3, use_container_width=True, key="ctx_trend")

                st.markdown(disclaimer(
                    "These charts reflect historical patterns in the Infoveave sample QC dataset. "
                    "They illustrate observed associations between defect types and production lines. "
                    "They do not establish causal relationships and should not replace process investigation."
                ), unsafe_allow_html=True)

    # ── Tab B4: Contributing Factors & Actions
    with inner[3]:
        if pred_cls == "normal":
            st.success(
                "✅ Model predicted **Normal** — no defect detected. "
                "No corrective action is indicated."
            )
        else:
            related_qc = VISION_TO_QC.get(pred_cls, [])
            if qc_df is not None and related_qc:
                for qc_type in related_qc:
                    st.markdown(
                        section_heading(f"Observed Associations — {qc_type}"),
                        unsafe_allow_html=True,
                    )
                    analysis = association_analysis(qc_df, qc_type)
                    if analysis["count"] == 0:
                        st.info(f"No records for '{qc_type}' in the dataset.")
                    else:
                        st.markdown(f"""
<div style="background:{COLORS['bg3']};border-radius:8px;padding:10px 14px;
            margin-bottom:12px;font-size:0.84rem;color:{COLORS['text2']};">
  Found <strong style="color:{COLORS['text']};">{analysis['count']} records</strong>
  ({analysis.get('freq_pct', 0):.1f}% of all defect records in the dataset)
</div>
""", unsafe_allow_html=True)
                        for obs in analysis["observations"]:
                            st.markdown(
                                finding_block(
                                    obs["finding"],
                                    obs["label"],
                                    obs["interpretation"],
                                    obs.get("validation", ""),
                                ),
                                unsafe_allow_html=True,
                            )

            # Corrective Action Guidance
            st.markdown("---")
            for qc_type in (related_qc if related_qc else [pred_cls.capitalize()]):
                guidance = corrective_action_guidance(qc_type)
                st.markdown(
                    section_heading(f"Suggested Investigation — {qc_type}"),
                    unsafe_allow_html=True,
                )

                immediate_html = "".join(
                    f"<div style='margin:4px 0;font-size:0.88rem;color:{COLORS['text']};'>"
                    f"<span style='color:{COLORS['reject']};font-weight:700;'>▶ {i}.</span> {s}</div>"
                    for i, s in enumerate(guidance["immediate_actions"], 1)
                )
                st.markdown(f"""
<div class="ms-action-card ms-action-immediate">
  <div class="ms-action-title">🚨 Immediate Actions</div>
  {immediate_html}
</div>""", unsafe_allow_html=True)

                invest_html = "".join(
                    f"<div style='margin:4px 0;font-size:0.88rem;color:{COLORS['text']};'>"
                    f"<span style='color:{COLORS['blue']};font-weight:700;'>▶ {i}.</span> {s}</div>"
                    for i, s in enumerate(guidance["investigation_areas"], 1)
                )
                st.markdown(f"""
<div class="ms-action-card ms-action-investigate">
  <div class="ms-action-title">🔎 Investigation Areas</div>
  {invest_html}
</div>""", unsafe_allow_html=True)

                st.markdown(disclaimer(guidance["disclaimer"]), unsafe_allow_html=True)

    # ── footer
    st.markdown(f"""
<div class="ms-footer">
  Inspection ID: <strong>{insp_id}</strong> &nbsp;·&nbsp;
  Model: ResNet18 (99.11% held-out accuracy) &nbsp;·&nbsp;
  Advisory only — no machine control
</div>
""", unsafe_allow_html=True)
