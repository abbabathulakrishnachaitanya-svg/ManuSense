"""app/pages/about.py — Page 6: About / Methodology"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.badges import section_heading, disclaimer, real_badge
from app.components.styles import COLORS


def render() -> None:
    st.markdown(section_heading("ℹ️ About ManuSense"), unsafe_allow_html=True)
    st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-left:4px solid {COLORS['accent']};border-radius:0 10px 10px 0;
            padding:20px 24px;margin-bottom:20px;font-size:0.95rem;color:{COLORS['text']};
            line-height:1.8;">
  <strong style="font-size:1.05rem;">ManuSense</strong> moves manufacturing quality
  from isolated defect detection toward <em>contextual, evidence-based decision support</em>.<br><br>
  The system connects visual AI inspection with historical quality control data to help
  engineers investigate defects, identify associated patterns, and prioritise corrective action —
  without fabricating evidence or claiming causality from correlation.
</div>
""", unsafe_allow_html=True)

    tabs = st.tabs([
        "🏗️ Architecture",
        "🔬 Methodology",
        "⚖️ Responsible AI",
        "📦 Data Sources",
        "🚀 Demo Flow",
    ])

    with tabs[0]:
        st.markdown(section_heading("Application Architecture"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-radius:10px;padding:20px 24px;font-size:0.88rem;
            color:{COLORS['text']};line-height:2;">
  <strong>Stack:</strong> Python · PyTorch · Streamlit · Plotly · Pandas · Pillow<br>
  <strong>Vision model:</strong> ResNet18 (fine-tuned) + Grad-CAM explainability<br>
  <strong>QC analytics:</strong> Infoveave Product Quality Control sample dataset (944 records)<br>
  <strong>UI:</strong> Dark navy industrial theme, 6-page Streamlit multi-page app<br><br>

  <strong>Module structure:</strong><br>
  <code style="color:{COLORS['accent']};">app/pages/</code> — Inspection, Dashboard, Root-Cause, Data Explorer, Model Validation, About<br>
  <code style="color:{COLORS['accent']};">app/components/</code> — styles.py, charts.py, badges.py (shared UI primitives)<br>
  <code style="color:{COLORS['accent']};">app/data/qc_data.py</code> — CSV loader, normaliser, KPI engine, association analysis<br>
  <code style="color:{COLORS['accent']};">app/vision/</code> — predictor.py, preprocessing.py, explainability.py, train.py<br>
  <code style="color:{COLORS['accent']};">models/best_model.pt</code> — Trained ResNet18 checkpoint (42 MB, included in repo)<br>
  <code style="color:{COLORS['accent']};">data/product-quality-control.csv</code> — Infoveave QC sample dataset
</div>
""", unsafe_allow_html=True)

        st.markdown(section_heading("Integration Roadmap"), unsafe_allow_html=True)
        stages = [
            ("MVP (current)", "Standalone web app, manual image upload, CSV-based QC analytics"),
            ("Phase 2", "REST API wrapper around predictor.py for production line integration"),
            ("Phase 3", "Camera feed integration, real-time inference pipeline"),
            ("Phase 4", "MES/ERP linkage via Inspection ID, live process variable feeds"),
        ]
        for stage, desc in stages:
            st.markdown(f"""
<div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:10px;">
  <span style="min-width:130px;font-weight:700;color:{COLORS['accent']};
               font-size:0.82rem;">{stage}</span>
  <span style="color:{COLORS['text2']};font-size:0.86rem;">{desc}</span>
</div>""", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown(section_heading("Methodology"), unsafe_allow_html=True)
        sections = [
            ("Visual Classification",
             "ResNet18 fine-tuned on 12,000 grayscale surface images across 5 defect classes. "
             "The model answers: 'What does this image appear to contain?' — not a causal diagnosis."),
            ("Confidence-Aware Decision",
             "Three-state logic (ACCEPT / REJECT / REVIEW) derived from configurable thresholds. "
             "The prototype threshold is clearly labelled and not claimed to be production-validated. "
             "Uncertain predictions always route to REVIEW."),
            ("Grad-CAM Explanation",
             "Gradient-weighted Class Activation Mapping highlights image regions that contributed "
             "to the prediction. Described as model attention visualization — not ground-truth localisation."),
            ("QC Dataset Association Analysis",
             "Descriptive analysis of the Infoveave sample dataset (944 records). "
             "Calculates frequency, line distribution, product association, and time trends "
             "for each defect type. All findings labelled as observed associations — not proven causes."),
            ("Corrective-Action Guidance",
             "Generic, operationally reasonable investigation areas based on defect category. "
             "Derived from standard manufacturing quality practice. "
             "Labelled as 'Suggested Investigation Areas' — never 'Proven Root Cause'."),
        ]
        for title, body in sections:
            st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-radius:8px;padding:14px 18px;margin-bottom:10px;">
  <div style="font-weight:700;color:{COLORS['text']};margin-bottom:5px;">{title}</div>
  <div style="font-size:0.86rem;color:{COLORS['text2']};line-height:1.7;">{body}</div>
</div>""", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown(section_heading("Responsible AI Commitments"), unsafe_allow_html=True)
        commitments = [
            ("✅", "Evidence is never fabricated",
             "All metrics calculated from source data. No hard-coded KPI values."),
            ("✅", "Predictions are never suppressed",
             "The model's output is always shown. Uncertain predictions surface as REVIEW, not hidden."),
            ("✅", "Causality is never claimed from correlation",
             "Historical associations are labelled as 'observed patterns' requiring process validation."),
            ("✅", "Human-in-the-loop for uncertain cases",
             "REVIEW state routes uncertain predictions to human inspection before any action."),
            ("✅", "Transparency about limitations",
             "Known failure modes (rust→hole confusion), domain shift risks, and data gaps are documented."),
            ("✅", "No fabricated factory results",
             "No invented customers, ROI figures, accuracy claims beyond the held-out test set."),
            ("✅", "Advisory only",
             "No machine control is performed. All decisions are recommendations to human operators."),
            ("✅", "Data quality disclosed",
             "Missing values, normalisation steps, and QC result inconsistencies are surfaced in the UI."),
        ]
        for icon, title, body in commitments:
            st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start;
            padding:10px 14px;background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-radius:8px;margin-bottom:8px;">
  <span style="font-size:1.1rem;margin-top:1px;">{icon}</span>
  <div>
    <div style="font-weight:700;color:{COLORS['text']};font-size:0.88rem;">{title}</div>
    <div style="color:{COLORS['text2']};font-size:0.83rem;margin-top:2px;">{body}</div>
  </div>
</div>""", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown(section_heading("Data Sources"), unsafe_allow_html=True)
        rows = [
            ("Computer vision images", "Challenge dataset — 12,000 images, 5 classes", "Real (challenge-provided)", "Model training & evaluation"),
            ("Trained model checkpoint", "ResNet18 fine-tuned weights (best_model.pt, 42 MB)", "Real", "Inference & Grad-CAM"),
            ("Test metrics", "Held-out evaluation (reports/test_metrics.json)", "Real", "Model Validation page"),
            ("Grad-CAM examples", "Generated from training images using trained model", "Real", "Reference gallery"),
            ("QC inspection records", "Infoveave Product Quality Control sample dataset (944 records)", "Sample / demonstration", "Dashboard, Root-Cause, Explorer"),
            ("Process variables", "NOT included — no temperature/pressure/vibration data in dataset", "N/A", "Not fabricated"),
            ("Economics figures", "NOT included — no real financial data provided", "N/A", "Not fabricated"),
        ]
        import pandas as pd
        st.dataframe(
            pd.DataFrame(rows, columns=["Data", "Description", "Type", "Used for"]),
            use_container_width=True, hide_index=True,
        )
        st.markdown(disclaimer(
            "The Infoveave Product Quality Control dataset is a publicly available sample dataset "
            "used here for demonstration purposes. It does not represent any real factory's production data."
        ), unsafe_allow_html=True)

    with tabs[4]:
        st.markdown(section_heading("2–3 Minute Demo Flow"), unsafe_allow_html=True)
        steps = [
            ("1", "Open Inspection tab",
             "Show the upload interface and explain the two-layer concept: vision model + QC dataset."),
            ("2", "Upload a surface image",
             "Use any of the Grad-CAM examples in reports/gradcam/ or any surface texture image."),
            ("3", "View AI prediction + decision",
             "Show the model prediction, confidence gauge, and the clear separation between "
             "'AI Model Prediction' and 'Operational Decision'."),
            ("4", "Open Model Attention tab",
             "Show Grad-CAM — explain it highlights where the model focused, not a defect mask."),
            ("5", "Open QC Historical Context tab",
             "Show how the visual prediction connects to the QC dataset — "
             "related defect types highlighted, line distribution, monthly trend."),
            ("6", "Open Contributing Factors tab",
             "Walk through the observed associations and suggested investigation areas."),
            ("7", "Switch to Quality Dashboard",
             "Show the executive overview — KPIs from the real CSV, defect distribution, trends."),
            ("8", "Switch to Root-Cause Investigation",
             "Select a defect type, apply filters — show association analysis and corrective guidance."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
<div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:10px;">
  <span style="min-width:28px;height:28px;background:{COLORS['accent']};color:white;
               border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
               font-weight:800;font-size:0.85rem;flex-shrink:0;">{num}</span>
  <div>
    <div style="font-weight:700;color:{COLORS['text']};font-size:0.88rem;">{title}</div>
    <div style="color:{COLORS['text2']};font-size:0.83rem;margin-top:2px;">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)
