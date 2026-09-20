"""app/pages/model_validation.py — Page 5: Model Validation"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.badges import section_heading, real_badge, disclaimer
from app.components.charts import confusion_matrix_chart
from app.components.styles import COLORS

GRADCAM_DIR = PROJECT_ROOT / "reports" / "gradcam"


def render() -> None:
    st.markdown(section_heading("📐 Model Validation", real_badge()), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{COLORS["text2"]};font-size:0.88rem;margin-top:-8px;margin-bottom:16px;">'
        "Held-out test-set performance for the ResNet18 defect classifier. "
        "All metrics are from a fixed held-out test split — never used during training.</p>",
        unsafe_allow_html=True,
    )

    st.markdown(disclaimer(
        "These benchmarks reflect performance on the prototype's held-out test set (15% stratified split, "
        "seed 42). They should NOT be interpreted as validated real-world factory performance. "
        "Out-of-distribution images (different lighting, resolution, or surface type) may produce "
        "lower accuracy."
    ), unsafe_allow_html=True)

    tabs = st.tabs([
        "📊 Performance Summary",
        "🧩 Confusion Matrix",
        "❌ Error Analysis",
        "🖼️ Grad-CAM Gallery",
        "⚙️ Architecture & Dataset",
    ])

    # ── Tab 1: Performance Summary
    with tabs[0]:
        st.markdown(section_heading("Held-Out Test Performance"), unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy",      "99.11%")
        m2.metric("Correct / 1,800",    "1,784")
        m3.metric("Incorrect / 1,800",  "16")
        m4.metric("Weighted F1",        "99.11%")

        st.markdown(section_heading("Per-Class Metrics"), unsafe_allow_html=True)
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

        st.markdown(section_heading("Dataset Split"), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "Set":      ["Train",        "Validation",    "Test (held-out)"],
                "Images":   ["8,400",        "1,800",         "1,800"],
                "Strategy": ["70% stratified","15% stratified","15% stratified"],
                "Used for": ["Training","Early stopping / selection","Final evaluation only"],
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption("Stratified split · random seed 42 · no data leakage · test set never seen during training")

    # ── Tab 2: Confusion Matrix
    with tabs[1]:
        confusion_matrix_chart(key="mv_cm")
        st.caption(
            "Rows = true class · Columns = predicted class · "
            "All errors concentrated in the rust row (rust misclassified as hole/normal)"
        )

    # ── Tab 3: Error Analysis
    with tabs[2]:
        st.markdown(section_heading("Error Analysis — 16 Incorrect Predictions"), unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-radius:10px;padding:18px 20px;margin-bottom:16px;">
  <div style="font-size:0.82rem;color:{COLORS['text2']};margin-bottom:10px;">
    Of 1,800 held-out test images, <strong style="color:{COLORS['text']};">16 were misclassified</strong>
    — all from the <strong style="color:{COLORS['accent']};">rust class</strong>.
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:0.86rem;">
    <tr style="border-bottom:1px solid {COLORS['border']};">
      <th style="text-align:left;padding:8px;color:{COLORS['text2']};">True Class</th>
      <th style="text-align:left;padding:8px;color:{COLORS['text2']};">Predicted As</th>
      <th style="text-align:right;padding:8px;color:{COLORS['text2']};">Count</th>
      <th style="text-align:left;padding:8px;color:{COLORS['text2']};">% of Rust Test Set</th>
    </tr>
    <tr style="border-bottom:1px solid {COLORS['border']};">
      <td style="padding:8px;color:{COLORS['text']};">rust</td>
      <td style="padding:8px;color:{COLORS['review']};">hole</td>
      <td style="text-align:right;padding:8px;font-weight:700;color:{COLORS['text']};">14</td>
      <td style="padding:8px;color:{COLORS['text2']};">3.9%</td>
    </tr>
    <tr>
      <td style="padding:8px;color:{COLORS['text']};">rust</td>
      <td style="padding:8px;color:{COLORS['review']};">normal</td>
      <td style="text-align:right;padding:8px;font-weight:700;color:{COLORS['text']};">2</td>
      <td style="padding:8px;color:{COLORS['text2']};">0.6%</td>
    </tr>
  </table>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="background:{COLORS['bg3']};border:1px solid {COLORS['border']};
            border-left:4px solid {COLORS['review']};border-radius:0 8px 8px 0;
            padding:14px 18px;font-size:0.86rem;color:{COLORS['text2']};">
  <strong style="color:{COLORS['text']};">Why rust → hole?</strong><br>
  Rust (oxidation patches) and hole defects share similar visual features in greyscale surface images —
  both appear as irregular dark regions with textured boundaries. The model's primary
  confusion mode is this visual similarity, not a systematic failure.<br><br>
  <strong style="color:{COLORS['text']};">Mitigation in ManuSense:</strong>
  When the model predicts <em>hole</em> with rust probability &gt; 15%,
  the UI surfaces a targeted warning. All uncertain predictions are routed to
  REVIEW rather than automatic rejection.
</div>
""", unsafe_allow_html=True)

        st.markdown(disclaimer(
            "Error statistics are derived from the held-out test set only. "
            "No additional error statistics have been fabricated or extrapolated."
        ), unsafe_allow_html=True)

    # ── Tab 4: Grad-CAM Gallery
    with tabs[3]:
        st.markdown(section_heading("Training Example Grad-CAM (Model Attention Visualizations)"),
                    unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:{COLORS["text2"]};font-size:0.84rem;margin-bottom:16px;">'
            "Generated from representative images in the training dataset using the trained model. "
            "Warm colours = regions that most influenced the prediction. "
            "This is a model explanation aid — not ground-truth defect localisation.</p>",
            unsafe_allow_html=True,
        )
        classes = ["crack", "hole", "normal", "rust", "scratch"]
        cols = st.columns(len(classes), gap="small")
        for i, cls in enumerate(classes):
            path = GRADCAM_DIR / f"gradcam_{cls}.png"
            with cols[i]:
                if path.exists():
                    from PIL import Image
                    st.image(Image.open(path), use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="height:160px;background:{COLORS["bg3"]};'
                        f'border-radius:8px;display:flex;align-items:center;'
                        f'justify-content:center;color:{COLORS["text2"]};font-size:0.78rem;">'
                        f'Image not found</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div style="text-align:center;font-size:0.8rem;font-weight:700;'
                    f'color:{COLORS["accent"]};margin-top:6px;">{cls.upper()}</div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 5: Architecture
    with tabs[4]:
        st.markdown(section_heading("Model Architecture"), unsafe_allow_html=True)
        st.markdown(f"""
<div style="background:{COLORS['bg2']};border:1px solid {COLORS['border']};
            border-radius:10px;padding:18px 20px;font-size:0.88rem;
            color:{COLORS['text']};line-height:1.9;">
  <strong>Architecture:</strong> ResNet18 (He et al., 2016) — 11M parameters<br>
  <strong>Pre-training:</strong> ImageNet weights (torchvision default)<br>
  <strong>Fine-tuning strategy:</strong> All layers unfrozen — full fine-tuning<br>
  <strong>Input:</strong> Grayscale images converted to 3-channel (channel repeat), 224×224<br>
  <strong>Output head:</strong> Linear(512 → 5) replacing the original FC layer<br>
  <strong>Loss:</strong> CrossEntropyLoss<br>
  <strong>Optimizer:</strong> Adam, lr=0.001<br>
  <strong>Batch size:</strong> 64 · <strong>Epochs:</strong> 5 (best checkpoint saved by val accuracy)<br>
  <strong>Augmentation:</strong> RandomHorizontalFlip + RandomRotation(10°) — training only<br>
  <strong>Normalisation:</strong> ImageNet mean/std (matches pretrained weights)<br>
  <strong>Device:</strong> CPU-compatible (no GPU required)
</div>
""", unsafe_allow_html=True)

        st.markdown(section_heading("Dataset"), unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame({
                "Property":   ["Total images", "Classes", "Images per class",
                                "Train", "Validation", "Test", "Random seed",
                                "Image format", "Image size"],
                "Value":      ["12,000", "5", "2,400",
                                "8,400 (70%)", "1,800 (15%)", "1,800 (15%)", "42",
                                "Grayscale PNG", "224×224 (after resize)"],
            }),
            use_container_width=True, hide_index=True,
        )
