"""app/pages/data_explorer.py — Page 4: Data Explorer + Quality Report"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.badges import section_heading, real_badge, synth_badge, disclaimer
from app.components.styles import COLORS
from app.data.qc_data import load_normalised, load_raw, data_quality_report, apply_filters


def render() -> None:
    st.markdown(section_heading("🗂️ Data Explorer", real_badge()), unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{COLORS["text2"]};font-size:0.88rem;margin-top:-8px;margin-bottom:16px;">'
        "Explore the raw QC dataset and review data-quality issues. "
        "ManuSense is transparent about missing values and normalisation steps.</p>",
        unsafe_allow_html=True,
    )

    try:
        df_norm = load_normalised()
        df_raw  = load_raw()
    except Exception as exc:
        st.error(f"Could not load dataset: {exc}")
        return

    # ── Data Quality Report
    st.markdown(section_heading("Data Quality Report"), unsafe_allow_html=True)
    dq = data_quality_report()

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Records",       dq["total_records"])
    q2.metric("Usable Records",      dq["usable_records"])
    q3.metric("Missing Defect Type", f"{dq['defect_type_missing']} ({dq['defect_type_pct_missing']}%)")
    q4.metric("Null QC Result",      dq["qc_result_null"])

    with st.expander("🔍 Full Data Quality Details", expanded=False):
        st.markdown(f"""
<div style="font-size:0.86rem;color:{COLORS['text']};line-height:1.8;">

**Missing / null counts:**
- `units_inspected` null: **{dq['numeric_nulls']['units_inspected']}**
- `units_passed` null: **{dq['numeric_nulls']['units_passed']}**
- `defect_count` null: **{dq['numeric_nulls']['defect_count']}**
- `qc_result` null: **{dq['qc_result_null']}**
- `defect_type` null: **{dq['defect_type_missing']}** ({dq['defect_type_pct_missing']}%) — most records with 0 defects have no type recorded
- `inspection_date` parse errors: **{dq['date_parse_errors']}**

**Mixed-case QC results found and normalised:** {dq['qc_result_mixed_case']} records

**Normalisations applied:**
</div>
""", unsafe_allow_html=True)
        for note in dq["normalisations_applied"]:
            st.markdown(f"- {note}")

        st.markdown(disclaimer(
            "The original CSV is never modified. Normalisations are applied in-memory "
            "for display and analysis only. Raw data is shown in the 'Raw Data' tab below."
        ), unsafe_allow_html=True)

    st.divider()

    # ── Explorer tabs
    tab_norm, tab_raw, tab_stats = st.tabs([
        "📋 Normalised View", "🗃️ Raw Data", "📈 Column Statistics"
    ])

    # filters
    lines_all    = sorted(df_norm["line_id"].dropna().unique().tolist())
    defects_all  = ["(all)"] + sorted(df_norm["defect_type"].dropna().unique().tolist())
    qc_all       = sorted(df_norm["qc_result_norm"].dropna().unique().tolist())

    with tab_norm:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_lines = st.multiselect("Filter: Line", lines_all, key="de_lines")
        with fc2:
            f_defect = st.selectbox("Filter: Defect Type", defects_all, key="de_defect")
        with fc3:
            f_qc = st.multiselect("Filter: QC Result", qc_all, key="de_qc")

        df_show = apply_filters(
            df_norm,
            lines=f_lines or None,
            defect_types=[f_defect] if f_defect != "(all)" else None,
            qc_results=f_qc or None,
        )

        # Select display columns
        display_cols = [
            "batch_id", "date", "line_id", "product_code",
            "units_inspected", "units_passed", "defect_count",
            "defect_type", "inspector_id", "qc_result_norm",
        ]
        available = [c for c in display_cols if c in df_show.columns]
        df_disp = df_show[available].copy()
        df_disp.columns = [c.replace("_norm", " (normalised)").replace("_", " ").title()
                           for c in available]

        st.caption(f"Showing {len(df_disp):,} of {len(df_norm):,} records")
        st.dataframe(df_disp, use_container_width=True, height=420, hide_index=True)

        csv_bytes = df_disp.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Filtered CSV",
            data=csv_bytes,
            file_name="manusense_qc_filtered.csv",
            mime="text/csv",
        )

    with tab_raw:
        st.caption(f"Original CSV — {len(df_raw):,} rows, {len(df_raw.columns)} columns. No modifications.")
        st.dataframe(df_raw, use_container_width=True, height=420, hide_index=True)

    with tab_stats:
        st.markdown(f'<div class="ms-section">Numeric Column Distributions</div>',
                    unsafe_allow_html=True)
        num_cols = ["units_inspected", "units_passed", "defect_count"]
        for col in num_cols:
            vals = df_norm[col].dropna()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric(f"{col} — mean",   f"{vals.mean():.1f}")
            c2.metric("median",          f"{vals.median():.1f}")
            c3.metric("min",             f"{vals.min():.0f}")
            c4.metric("max",             f"{vals.max():.0f}")
            c5.metric("null count",      f"{df_norm[col].isna().sum()}")

        st.divider()
        st.markdown(f'<div class="ms-section">Categorical Distributions</div>',
                    unsafe_allow_html=True)

        cat1, cat2 = st.columns(2)
        with cat1:
            st.markdown("**QC Result (normalised)**")
            st.dataframe(
                df_norm["qc_result_norm"].value_counts(dropna=False).reset_index()
                .rename(columns={"qc_result_norm": "QC Result", "count": "Count"}),
                use_container_width=True, hide_index=True,
            )
        with cat2:
            st.markdown("**Defect Type**")
            st.dataframe(
                df_norm["defect_type"].value_counts(dropna=False).reset_index()
                .rename(columns={"defect_type": "Defect Type", "count": "Count"}),
                use_container_width=True, hide_index=True,
            )
