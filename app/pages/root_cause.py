"""app/pages/root_cause.py — Page 3: Root-Cause Investigation"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.badges import (
    section_heading, real_badge, disclaimer,
    finding_block,
)
from app.components.styles import COLORS, LINE_COLORS, DEFECT_COLORS
from app.data.qc_data import (
    load_normalised, apply_filters,
    association_analysis, corrective_action_guidance,
    defects_by_line, defect_type_by_line, trend_by_month,
)


def _chart(fig, key, h=280):
    fig.update_layout(
        height=h, paper_bgcolor=COLORS["bg2"], plot_bgcolor=COLORS["bg2"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=14, r=14, t=36, b=12),
        xaxis=dict(gridcolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"]),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def render() -> None:
    st.markdown(section_heading("🔎 Root-Cause Investigation", real_badge()), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{COLORS["text2"]};font-size:0.88rem;margin-top:-8px;margin-bottom:16px;">'
        "Select a defect type and apply filters to explore observed patterns in the QC dataset. "
        "All findings are descriptive associations — not proven causal relationships.</p>",
        unsafe_allow_html=True,
    )

    try:
        df_full = load_normalised()
    except Exception as exc:
        st.error(f"Could not load QC dataset: {exc}")
        return

    defect_types = sorted(df_full["defect_type"].dropna().unique().tolist())
    lines_all    = sorted(df_full["line_id"].dropna().unique().tolist())
    products_all = sorted(df_full["product_code"].dropna().unique().tolist())

    # ── Controls
    st.markdown(section_heading("Investigation Controls"), unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        sel_defect = st.selectbox("Defect Type", defect_types, key="rc_defect")
    with ctrl2:
        sel_lines = st.multiselect("Production Line", lines_all, key="rc_lines")
    with ctrl3:
        sel_qc = st.multiselect("QC Result", ["PASS", "FAIL", "PENDING"], key="rc_qc_filter")
    with ctrl4:
        years = sorted(df_full["year"].dropna().unique().astype(int).tolist())
        if years:
            yr = st.select_slider("Year Range", options=years,
                                  value=(min(years), max(years)), key="rc_year")
        else:
            yr = None

    date_start = f"{yr[0]}-01-01" if yr else None
    date_end   = f"{yr[1]}-12-31" if yr else None

    df = apply_filters(df_full,
                       lines=sel_lines or None,
                       qc_results=sel_qc or None,
                       date_start=date_start,
                       date_end=date_end)

    df_defect = df[df["defect_type"] == sel_defect]

    st.divider()

    if df_defect.empty:
        st.info(f"No records found for **{sel_defect}** with the current filters.")
        return

    # ── Summary strip
    total_records      = len(df.dropna(subset=["defect_type"]))
    defect_count       = len(df_defect)
    freq_pct           = defect_count / total_records * 100 if total_records else 0

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Records (filtered)", f"{len(df):,}")
    s2.metric(f"{sel_defect} count", defect_count)
    s3.metric("% of defect records", f"{freq_pct:.1f}%")
    s4.metric("Lines affected",
              df_defect["line_id"].nunique())

    st.divider()

    # ── Row 1: line + monthly trend
    st.markdown(section_heading("Distribution Analysis"), unsafe_allow_html=True)
    r1a, r1b = st.columns(2, gap="large")

    with r1a:
        line_counts = df_defect["line_id"].value_counts().reset_index()
        line_counts.columns = ["line_id", "count"]
        colors = [LINE_COLORS.get(l, COLORS["accent"]) for l in line_counts["line_id"]]
        fig = go.Figure(go.Bar(
            x=line_counts["line_id"], y=line_counts["count"],
            marker_color=colors,
            text=line_counts["count"], textposition="outside",
        ))
        fig.update_layout(title=f"{sel_defect} by Production Line",
                          xaxis_title="Line", yaxis_title="Count")
        _chart(fig, "rc_line")

    with r1b:
        monthly = df_defect.dropna(subset=["date"]).groupby("year_month").size().reset_index(name="count")
        monthly = monthly.sort_values("year_month")
        fig2 = go.Figure(go.Scatter(
            x=monthly["year_month"], y=monthly["count"],
            mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy", fillcolor=f"rgba(232,80,10,0.12)",
            marker=dict(size=5),
        ))
        fig2.update_layout(title=f"Monthly Trend — {sel_defect}",
                           xaxis_title="Month", yaxis_title="Count")
        _chart(fig2, "rc_trend")

    # ── Row 2: product + QC result
    st.markdown(section_heading("Product & QC Analysis"), unsafe_allow_html=True)
    r2a, r2b = st.columns(2, gap="large")

    with r2a:
        prod_counts = df_defect["product_code"].value_counts().head(10).reset_index()
        prod_counts.columns = ["product_code", "count"]
        fig3 = go.Figure(go.Bar(
            x=prod_counts["product_code"], y=prod_counts["count"],
            marker_color=COLORS["accent"],
            text=prod_counts["count"], textposition="outside",
        ))
        fig3.update_layout(title=f"Top Products — {sel_defect}",
                           xaxis_title="Product Code", yaxis_title="Count",
                           xaxis_tickangle=-35)
        _chart(fig3, "rc_prod")

    with r2b:
        qc_dist = df_defect["qc_result_norm"].value_counts(dropna=True).reset_index()
        qc_dist.columns = ["qc_result_norm", "count"]
        color_map = {"PASS": COLORS["accept"], "FAIL": COLORS["reject"],
                     "PENDING": COLORS["review"]}
        qc_colors = [color_map.get(r, COLORS["text2"]) for r in qc_dist["qc_result_norm"]]
        fig4 = go.Figure(go.Bar(
            x=qc_dist["qc_result_norm"], y=qc_dist["count"],
            marker_color=qc_colors,
            text=qc_dist["count"], textposition="outside",
        ))
        fig4.update_layout(title=f"QC Outcomes — {sel_defect}",
                           xaxis_title="QC Result", yaxis_title="Count")
        _chart(fig4, "rc_qc")

    # ── All lines comparison
    st.markdown(section_heading("Cross-Line Comparison"), unsafe_allow_html=True)
    cross = defect_type_by_line(df)
    if not cross.empty:
        sel_cross = cross[cross["defect_type"] == sel_defect]
        if not sel_cross.empty:
            lc = [LINE_COLORS.get(l, COLORS["accent"]) for l in sel_cross["line_id"]]
            fig5 = go.Figure(go.Bar(
                x=sel_cross["line_id"], y=sel_cross["count"],
                marker_color=lc, text=sel_cross["count"], textposition="outside",
            ))
            fig5.update_layout(title=f"{sel_defect} — per line (full dataset)",
                               xaxis_title="Line", yaxis_title="Count")
            _chart(fig5, "rc_cross", 240)

    st.divider()

    # ── Association observations
    st.markdown(section_heading("Observed Associations"), unsafe_allow_html=True)
    analysis = association_analysis(df_full, sel_defect)
    if analysis["observations"]:
        for obs in analysis["observations"]:
            st.markdown(
                finding_block(obs["finding"], obs["label"],
                              obs["interpretation"], obs.get("validation", "")),
                unsafe_allow_html=True,
            )
    else:
        st.info("No significant associations found for the current selection.")

    st.divider()

    # ── Corrective action guidance
    st.markdown(section_heading("Suggested Investigation Areas"), unsafe_allow_html=True)
    guidance = corrective_action_guidance(sel_defect)

    imm_html = "".join(
        f"<div style='margin:4px 0;font-size:0.88rem;color:{COLORS['text']};'>"
        f"<span style='color:{COLORS['reject']};font-weight:700;'>▶ {i}.</span> {s}</div>"
        for i, s in enumerate(guidance["immediate_actions"], 1)
    )
    st.markdown(f"""
<div class="ms-action-card ms-action-immediate">
  <div class="ms-action-title">🚨 Immediate Actions</div>{imm_html}
</div>""", unsafe_allow_html=True)

    inv_html = "".join(
        f"<div style='margin:4px 0;font-size:0.88rem;color:{COLORS['text']};'>"
        f"<span style='color:{COLORS['blue']};font-weight:700;'>▶ {i}.</span> {s}</div>"
        for i, s in enumerate(guidance["investigation_areas"], 1)
    )
    st.markdown(f"""
<div class="ms-action-card ms-action-investigate">
  <div class="ms-action-title">🔎 Investigation Areas</div>{inv_html}
</div>""", unsafe_allow_html=True)

    st.markdown(disclaimer(guidance["disclaimer"]), unsafe_allow_html=True)
