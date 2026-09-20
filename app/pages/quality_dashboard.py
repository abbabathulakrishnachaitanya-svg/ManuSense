"""app/pages/quality_dashboard.py — Page 2: Quality Dashboard"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.badges import section_heading, synth_badge, real_badge, disclaimer
from app.components.charts import (
    defect_bar, defect_pie, line_defect_bar, line_defect_rate,
    trend_line, qc_pass_fail, defect_heatmap, product_defect_bar,
)
from app.components.styles import COLORS
from app.data.qc_data import (
    load_normalised, compute_kpis, defects_by_type,
    defects_by_line, trend_by_month, defect_type_by_line,
    qc_result_by_line, defects_by_product, apply_filters,
)


def render() -> None:
    st.markdown(section_heading("📊 Quality Dashboard", real_badge()), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{COLORS["text2"]};font-size:0.88rem;margin-top:-8px;margin-bottom:16px;">'
        "Executive overview — all metrics calculated from the Infoveave Product Quality Control "
        "sample dataset (944 inspection records, 2022–2024). No values are hard-coded.</p>",
        unsafe_allow_html=True,
    )

    try:
        df_full = load_normalised()
    except Exception as exc:
        st.error(f"Could not load QC dataset: {exc}")
        return

    # ── Sidebar filters
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🔽 Dashboard Filters**")
        lines = st.multiselect(
            "Production Line", sorted(df_full["line_id"].dropna().unique()),
            key="dash_lines",
        )
        qc_choices = st.multiselect(
            "QC Result", ["PASS", "FAIL", "PENDING"],
            key="dash_qc",
        )
        years = sorted(df_full["year"].dropna().unique().astype(int).tolist())
        year_range = st.select_slider(
            "Year", options=years, value=(min(years), max(years)),
            key="dash_year",
        ) if years else None

    date_start = f"{year_range[0]}-01-01" if year_range else None
    date_end   = f"{year_range[1]}-12-31" if year_range else None

    df = apply_filters(df_full, lines=lines or None, qc_results=qc_choices or None,
                       date_start=date_start, date_end=date_end)

    if df.empty:
        st.warning("No records match the current filters.")
        return

    kpis = compute_kpis(df)

    # ── KPI cards row 1
    st.markdown(section_heading("Key Performance Indicators"), unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Inspections",  f"{kpis['total_inspections']:,}")
    k2.metric("Units Inspected",    f"{kpis['total_units']:,}")
    k3.metric("Total Defects",      f"{kpis['total_defects']:,}")
    k4.metric("Pass Rate",          f"{kpis['pass_rate_pct']:.1f}%")
    k5.metric("Defect Rate",        f"{kpis['defect_rate_pct']:.2f}%")

    # ── KPI cards row 2
    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric("QC PASS",     f"{kpis['qc_pass']:,}")
    k7.metric("QC FAIL",     f"{kpis['qc_fail']:,}")
    k8.metric("QC PENDING",  f"{kpis['qc_pending']:,}")
    k9.metric("Batch Fail Rate", f"{kpis['batch_fail_rate_pct']:.1f}%")
    k10.metric("Unknown QC", f"{kpis['qc_unknown']:,}")

    st.divider()

    # ── Row 1: defect type charts
    st.markdown(section_heading("Defect Type Analysis"), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        defect_bar(defects_by_type(df), key="dash_def_bar")
    with c2:
        defect_pie(defects_by_type(df), key="dash_def_pie")

    # ── Row 2: production line
    st.markdown(section_heading("Production Line Analysis"), unsafe_allow_html=True)
    c3, c4 = st.columns(2, gap="large")
    with c3:
        line_defect_bar(defects_by_line(df), key="dash_line_bar")
    with c4:
        line_defect_rate(defects_by_line(df), key="dash_line_rate")

    # ── Row 3: trend + QC pass/fail
    st.markdown(section_heading("Trends & QC Results"), unsafe_allow_html=True)
    c5, c6 = st.columns(2, gap="large")
    with c5:
        trend_line(trend_by_month(df), key="dash_trend")
    with c6:
        qc_pass_fail(qc_result_by_line(df), key="dash_qc_pf")

    # ── Row 4: heatmap + product
    st.markdown(section_heading("Cross-Analysis"), unsafe_allow_html=True)
    c7, c8 = st.columns(2, gap="large")
    with c7:
        defect_heatmap(defect_type_by_line(df), key="dash_heat")
    with c8:
        product_defect_bar(defects_by_product(df, top_n=15), key="dash_prod")

    st.markdown(disclaimer(
        "All metrics are calculated from the Infoveave Product Quality Control sample dataset. "
        "This is demonstration data and does not represent a real factory's production records."
    ), unsafe_allow_html=True)
