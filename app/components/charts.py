"""
app/components/charts.py
─────────────────────────
Reusable Plotly chart helpers for ManuSense.
All charts use the dark industrial theme from styles.py.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.components.styles import COLORS, DEFECT_COLORS, LINE_COLORS

_LAYOUT = dict(
    paper_bgcolor=COLORS["bg2"],
    plot_bgcolor=COLORS["bg2"],
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=16, r=16, t=36, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"])),
    xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], tickfont=dict(color=COLORS["text2"])),
    yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"], tickfont=dict(color=COLORS["text2"])),
)


def _apply(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, **_LAYOUT)
    return fig


def render(fig: go.Figure, key: str, height: int = 300) -> None:
    _apply(fig, height)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ─────────────────────────────────────────────────────────────────────────────

def defect_bar(df: pd.DataFrame, key: str = "defect_bar", height: int = 300) -> None:
    """Horizontal bar: defect count by type."""
    if df.empty:
        st.info("No defect type data available for the selected filters.")
        return
    colors = [DEFECT_COLORS.get(t, COLORS["accent"]) for t in df["defect_type"]]
    fig = go.Figure(go.Bar(
        x=df["count"], y=df["defect_type"],
        orientation="h",
        marker_color=colors,
        text=df["count"], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(title="Defect Count by Type",
                      xaxis_title="Count", yaxis_title="")
    render(fig, key, height)


def defect_pie(df: pd.DataFrame, key: str = "defect_pie", height: int = 300) -> None:
    """Donut chart: defect type distribution."""
    if df.empty:
        st.info("No defect data available.")
        return
    colors = [DEFECT_COLORS.get(t, COLORS["accent"]) for t in df["defect_type"]]
    fig = go.Figure(go.Pie(
        labels=df["defect_type"], values=df["count"],
        hole=0.45, marker_colors=colors,
        textinfo="label+percent",
    ))
    fig.update_layout(title="Defect Type Distribution",
                      legend=dict(orientation="h", y=-0.18))
    render(fig, key, height)


def line_defect_bar(df: pd.DataFrame, key: str = "line_bar", height: int = 300) -> None:
    """Bar: defect count by production line."""
    if df.empty:
        st.info("No data available.")
        return
    colors = [LINE_COLORS.get(l, COLORS["accent"]) for l in df["line_id"]]
    fig = go.Figure(go.Bar(
        x=df["line_id"], y=df["defect_count"],
        marker_color=colors,
        text=df["defect_count"], textposition="outside",
    ))
    fig.update_layout(title="Defect Count by Production Line",
                      xaxis_title="Line", yaxis_title="Defect Count")
    render(fig, key, height)


def line_defect_rate(df: pd.DataFrame, key: str = "line_rate", height: int = 300) -> None:
    """Bar: defect rate % by production line."""
    if df.empty:
        st.info("No data available.")
        return
    colors = [LINE_COLORS.get(l, COLORS["accent"]) for l in df["line_id"]]
    fig = go.Figure(go.Bar(
        x=df["line_id"],
        y=df["defect_rate_pct"],
        marker_color=colors,
        text=[f"{v:.2f}%" for v in df["defect_rate_pct"]],
        textposition="outside",
    ))
    fig.update_layout(title="Defect Rate (%) by Production Line",
                      xaxis_title="Line", yaxis_title="Defect Rate (%)")
    render(fig, key, height)


def trend_line(df: pd.DataFrame, key: str = "trend", height: int = 320) -> None:
    """Line chart: monthly defect count trend."""
    if df.empty:
        st.info("No trend data available.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["year_month"], y=df["defect_count"],
        mode="lines+markers", name="Defect Count",
        line=dict(color=COLORS["reject"], width=2),
        marker=dict(size=5),
    ))
    fig.add_trace(go.Bar(
        x=df["year_month"], y=df["inspections"],
        name="Inspections", yaxis="y2",
        marker_color=COLORS["blue"], opacity=0.3,
    ))
    fig.update_layout(
        title="Monthly Defect Trend",
        xaxis_title="Month",
        yaxis=dict(title="Defect Count", gridcolor=COLORS["border"]),
        yaxis2=dict(title="Inspections", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color=COLORS["text2"])),
        legend=dict(orientation="h", y=1.12),
    )
    render(fig, key, height)


def qc_pass_fail(df_line: pd.DataFrame, key: str = "qc_pf", height: int = 300) -> None:
    """Stacked bar: PASS/FAIL per line."""
    if df_line.empty:
        st.info("No QC result data.")
        return
    fig = px.bar(
        df_line, x="line_id", y="count", color="qc_result_norm",
        color_discrete_map={"PASS": COLORS["accept"], "FAIL": COLORS["reject"],
                            "PENDING": COLORS["review"]},
        barmode="stack",
        title="QC Result by Production Line",
        labels={"line_id": "Line", "count": "Count", "qc_result_norm": "QC Result"},
    )
    fig.update_layout(paper_bgcolor=COLORS["bg2"], plot_bgcolor=COLORS["bg2"],
                      font=dict(color=COLORS["text"]),
                      margin=dict(l=16, r=16, t=36, b=16),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, use_container_width=True, key=key)


def defect_heatmap(df_cross: pd.DataFrame, key: str = "heat", height: int = 320) -> None:
    """Heatmap: defect type × production line."""
    if df_cross.empty:
        st.info("No cross-tabulation data.")
        return
    pivot = df_cross.pivot(index="defect_type", columns="line_id", values="count").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="OrRd",
        text=pivot.values.astype(int),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(title="Defect Type × Production Line",
                      xaxis_title="Line", yaxis_title="Defect Type",
                      paper_bgcolor=COLORS["bg2"], plot_bgcolor=COLORS["bg2"],
                      font=dict(color=COLORS["text"]),
                      margin=dict(l=16, r=16, t=36, b=16))
    st.plotly_chart(fig, use_container_width=True, key=key)


def product_defect_bar(df: pd.DataFrame, key: str = "prod_bar", height: int = 320) -> None:
    """Bar: top products by defect count."""
    if df.empty:
        st.info("No product defect data.")
        return
    fig = go.Figure(go.Bar(
        x=df["product_code"], y=df["defect_count"],
        marker_color=COLORS["accent"],
        text=df["defect_count"], textposition="outside",
    ))
    fig.update_layout(title="Top Products by Defect Count",
                      xaxis_title="Product Code", yaxis_title="Defect Count",
                      xaxis_tickangle=-40)
    render(fig, key, height)


def confidence_gauge(confidence: float, decision: str, key: str = "gauge") -> None:
    """Gauge chart for model confidence."""
    color = COLORS["accept"] if decision == "ACCEPT" else \
            COLORS["reject"] if decision == "REJECT" else COLORS["review"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number=dict(suffix="%", font=dict(size=28, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=COLORS["text2"]),
            bar=dict(color=color),
            bgcolor=COLORS["bg3"],
            bordercolor=COLORS["border"],
            steps=[
                dict(range=[0, 50],  color=COLORS["bg3"]),
                dict(range=[50, 80], color="rgba(210,153,34,0.15)"),
                dict(range=[80, 100],color="rgba(35,134,54,0.15)"),
            ],
            threshold=dict(
                line=dict(color=COLORS["review"], width=2),
                thickness=0.75, value=80,
            ),
        ),
        title=dict(text="Confidence", font=dict(color=COLORS["text2"], size=13)),
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor=COLORS["bg2"],
        plot_bgcolor=COLORS["bg2"],
        margin=dict(l=20, r=20, t=40, b=10),
        font=dict(color=COLORS["text"]),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def prob_bar(probabilities: dict[str, float], predicted_class: str,
             key: str = "prob_bar") -> None:
    """Horizontal bar showing all class probabilities."""
    CLASS_COLORS = {
        "normal":  COLORS["accept"],
        "crack":   COLORS["reject"],
        "hole":    "#e8500a",
        "rust":    "#d29922",
        "scratch": COLORS["blue"],
    }
    sorted_p = sorted(probabilities.items(), key=lambda x: -x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in sorted_p],
        y=[c.capitalize() for c, _ in sorted_p],
        orientation="h",
        marker_color=[CLASS_COLORS.get(c, COLORS["accent"]) for c, _ in sorted_p],
        text=[f"{v:.1%}" for _, v in sorted_p],
        textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Class Probabilities",
        xaxis=dict(range=[0, 1.15], tickformat=".0%", title=""),
        yaxis=dict(title=""),
    )
    render(fig, key, 220)


def confusion_matrix_chart(key: str = "cm") -> None:
    """Static confusion matrix from held-out test results."""
    import numpy as np
    cm = [[360, 0, 0, 0, 0],
          [0, 360, 0, 0, 0],
          [0, 0, 360, 0, 0],
          [0, 14, 2, 344, 0],
          [0, 0, 0, 0, 360]]
    classes = ["crack", "hole", "normal", "rust", "scratch"]
    fig = px.imshow(
        cm, x=classes, y=classes,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Predicted", y="True", color="Count"),
        title="Confusion Matrix — Held-Out Test Set (1,800 images)",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor=COLORS["bg2"],
        plot_bgcolor=COLORS["bg2"],
        font=dict(color=COLORS["text"]),
        margin=dict(l=16, r=16, t=48, b=16),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
