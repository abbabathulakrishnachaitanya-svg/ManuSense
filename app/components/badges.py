"""
app/components/badges.py
─────────────────────────
HTML badge/label helpers for ManuSense UI.
All return strings safe to use with st.markdown(..., unsafe_allow_html=True).
"""
from __future__ import annotations


def decision_badge(decision: str) -> str:
    cls = {
        "ACCEPT": "ms-badge-accept",
        "REJECT": "ms-badge-reject",
        "REVIEW": "ms-badge-review",
    }.get(decision, "ms-badge-review")
    icons = {"ACCEPT": "✅", "REJECT": "❌", "REVIEW": "🔍"}
    icon = icons.get(decision, "⚠️")
    return f'<span class="{cls}">{icon} {decision}</span>'


def qc_badge(result: str) -> str:
    upper = str(result).upper().strip()
    cls = "ms-badge-pass" if upper == "PASS" else \
          "ms-badge-fail" if upper == "FAIL" else "ms-badge-info"
    return f'<span class="{cls}">{upper}</span>'


def label_badge(text: str, style: str = "info") -> str:
    cls = f"ms-badge-{style}"
    return f'<span class="{cls}">{text}</span>'


def section_heading(text: str, badge: str = "") -> str:
    return f'<div class="ms-section">{text}{badge}</div>'


def synth_badge() -> str:
    return '<span class="ms-badge-synth">SAMPLE DATA</span>'


def real_badge() -> str:
    return '<span class="ms-badge-real">REAL DATA</span>'


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<span class="ms-kpi-sub">{sub}</span>' if sub else ""
    return f"""
<div class="ms-kpi">
  <span class="ms-kpi-label">{label}</span>
  <span class="ms-kpi-value">{value}</span>
  {sub_html}
</div>"""


def finding_block(finding: str, label: str, interpretation: str,
                  validation: str = "") -> str:
    val_html = f'<span class="ms-validation-note">🔬 {validation}</span>' if validation else ""
    return f"""
<div class="ms-finding">
  <strong style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.5px;
                  color:#e3b341;">{label}</strong><br>
  {finding}
  <span class="ms-interp">💡 {interpretation}</span>
  {val_html}
</div>"""


def disclaimer(text: str) -> str:
    return f'<div class="ms-disclaimer">ℹ️ {text}</div>'
