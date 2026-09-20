"""
app/components/styles.py
─────────────────────────
Shared CSS for ManuSense — dark navy industrial theme.
Call inject_css() once at the top of the main app file.
"""

import streamlit as st

# Colour tokens
C_BG        = "#0d1117"   # page background
C_BG2       = "#161b22"   # card / panel background
C_BG3       = "#21262d"   # input / table row background
C_BORDER    = "#30363d"   # borders
C_TEXT      = "#e6edf3"   # primary text
C_TEXT2     = "#8b949e"   # secondary / muted text
C_ACCENT    = "#e8500a"   # ManuSense orange
C_ACCEPT    = "#238636"   # green
C_REJECT    = "#da3633"   # red
C_REVIEW    = "#d29922"   # amber
C_BLUE      = "#1f6feb"   # info blue


def inject_css() -> None:
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Base ─────────────────────────────────────────────────── */
html, body, .stApp, .stApp > div,
section[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"] > div,
div[data-testid="stVerticalBlock"],
.block-container, .main {{
    background-color: {C_BG} !important;
    color: {C_TEXT} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {C_BG2} !important;
    border-right: 1px solid {C_BORDER} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {C_TEXT} !important;
}}

.block-container {{ padding: 1.4rem 2.4rem 2rem 2.4rem !important; max-width: 1320px; }}

/* ── Typography ───────────────────────────────────────────── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
label {{ color: {C_TEXT} !important; }}

/* ── Tabs ─────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background-color: {C_BG2} !important;
    border-bottom: 1px solid {C_BORDER} !important;
    gap: 2px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: {C_BG2} !important;
    color: {C_TEXT2} !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 6px 6px 0 0 !important;
    border-bottom: 2px solid transparent !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C_ACCENT} !important;
    border-bottom: 2px solid {C_ACCENT} !important;
    background-color: {C_BG} !important;
}}
[data-testid="stTabsContent"],
[data-testid="stTabsContent"] > div {{
    background-color: {C_BG} !important;
}}

/* ── Metrics ──────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background-color: {C_BG2} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
}}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"] div,
[data-testid="stMetricDelta"] {{ color: {C_TEXT} !important; }}

/* ── Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 6px !important;
}}
[data-testid="stAlert"] p,
[data-testid="stAlert"] div {{
    color: #1a1a1a !important;
}}

/* ── Expander ─────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    background-color: {C_BG2} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {{
    color: {C_TEXT} !important;
    font-weight: 600 !important;
}}
[data-testid="stExpander"] > div > div {{
    background-color: {C_BG2} !important;
    color: {C_TEXT} !important;
}}

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
    background-color: {C_BG2} !important;
    border: 2px dashed {C_BORDER} !important;
    border-radius: 10px !important;
    padding: 12px !important;
}}
[data-testid="stFileUploader"] * {{ color: {C_TEXT} !important; }}

/* ── Selectbox / inputs ───────────────────────────────────── */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {{
    background-color: {C_BG3} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
}}
.stTextInput input, .stNumberInput input {{
    background-color: {C_BG3} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
}}

/* ── DataFrame ────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    background-color: {C_BG2} !important;
}}
.dvn-scroller {{ background-color: {C_BG2} !important; }}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton button {{
    background-color: {C_ACCENT} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}}
.stButton button:hover {{
    background-color: #c9440a !important;
}}
.stDownloadButton button {{
    background-color: {C_BG3} !important;
    color: {C_TEXT} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 6px !important;
}}

/* ── Slider ───────────────────────────────────────────────── */
[data-testid="stSlider"] * {{ color: {C_TEXT} !important; }}
[data-baseweb="slider"] [data-testid="stThumbValue"] {{
    background-color: {C_ACCENT} !important;
}}

/* ── Divider ──────────────────────────────────────────────── */
hr {{ border-color: {C_BORDER} !important; }}

/* ── Caption ──────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p {{
    color: {C_TEXT2} !important;
    font-size: 0.78rem !important;
}}

/* ── Progress ─────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div {{
    background-color: {C_ACCENT} !important;
}}

/* ── Custom component classes ─────────────────────────────── */
.ms-topbar {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0 14px 0;
    border-bottom: 1px solid {C_BORDER};
    margin-bottom: 24px;
}}
.ms-wordmark {{
    font-size: 1.5rem; font-weight: 900;
    color: {C_TEXT} !important; letter-spacing: -0.4px;
}}
.ms-wordmark em {{
    font-style: normal; color: {C_ACCENT} !important;
}}
.ms-tagline {{
    font-size: 0.75rem; color: {C_TEXT2} !important;
    margin-top: 2px; font-style: italic;
}}

.ms-section {{
    font-size: 0.78rem; font-weight: 800; color: {C_TEXT} !important;
    text-transform: uppercase; letter-spacing: .7px;
    border-left: 3px solid {C_ACCENT}; padding-left: 9px;
    margin: 20px 0 10px 0; display: block;
    background: transparent !important;
}}

.ms-card {{
    background: {C_BG2}; border: 1px solid {C_BORDER};
    border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;
}}
.ms-card-accent {{
    border-left: 4px solid {C_ACCENT};
}}

.ms-kpi {{
    background: {C_BG2}; border: 1px solid {C_BORDER};
    border-radius: 10px; padding: 16px 18px;
    text-align: center;
}}
.ms-kpi-label {{
    font-size: 0.73rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .5px; color: {C_TEXT2}; display: block; margin-bottom: 6px;
}}
.ms-kpi-value {{
    font-size: 1.9rem; font-weight: 900; color: {C_TEXT}; display: block;
}}
.ms-kpi-sub {{
    font-size: 0.72rem; color: {C_TEXT2}; margin-top: 4px; display: block;
}}

.ms-badge-accept {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    background: rgba(35,134,54,0.2); color: #3fb950 !important;
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .5px; border: 1px solid {C_ACCEPT};
}}
.ms-badge-reject {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    background: rgba(218,54,51,0.2); color: #f85149 !important;
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .5px; border: 1px solid {C_REJECT};
}}
.ms-badge-review {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    background: rgba(210,153,34,0.2); color: #e3b341 !important;
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .5px; border: 1px solid {C_REVIEW};
}}
.ms-badge-pass {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    background: rgba(35,134,54,0.15); color: #3fb950 !important;
    font-size: 0.72rem; font-weight: 700; border: 1px solid {C_ACCEPT};
}}
.ms-badge-fail {{
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    background: rgba(218,54,51,0.15); color: #f85149 !important;
    font-size: 0.72rem; font-weight: 700; border: 1px solid {C_REJECT};
}}
.ms-badge-info {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    background: rgba(31,111,235,0.15); color: #58a6ff !important;
    font-size: 0.7rem; font-weight: 700; border: 1px solid {C_BLUE};
}}
.ms-badge-synth {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    background: rgba(210,153,34,0.15); color: #e3b341 !important;
    font-size: 0.68rem; font-weight: 700; border: 1px solid {C_REVIEW};
    margin-left: 6px;
}}
.ms-badge-real {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    background: rgba(35,134,54,0.15); color: #3fb950 !important;
    font-size: 0.68rem; font-weight: 700; border: 1px solid {C_ACCEPT};
    margin-left: 6px;
}}

.ms-finding {{
    background: {C_BG3}; border: 1px solid {C_BORDER};
    border-left: 4px solid {C_REVIEW};
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0;
    font-size: 0.86rem; color: {C_TEXT};
}}
.ms-interp {{
    font-size: 0.82rem; color: {C_TEXT2}; margin-top: 5px; display: block;
}}
.ms-validation-note {{
    font-size: 0.76rem; color: #58a6ff; margin-top: 4px; display: block;
    font-style: italic;
}}

.ms-action-card {{
    background: {C_BG3}; border: 1px solid {C_BORDER};
    border-left: 4px solid;
    border-radius: 0 8px 8px 0; padding: 14px 16px; margin: 6px 0;
}}
.ms-action-immediate  {{ border-left-color: {C_REJECT}; }}
.ms-action-investigate{{ border-left-color: {C_BLUE}; }}
.ms-action-correct    {{ border-left-color: {C_ACCEPT}; }}
.ms-action-title {{
    font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .6px; margin-bottom: 8px; display: block;
}}
.ms-action-immediate .ms-action-title  {{ color: #f85149; }}
.ms-action-investigate .ms-action-title{{ color: #58a6ff; }}
.ms-action-correct .ms-action-title    {{ color: #3fb950; }}

.ms-disclaimer {{
    background: rgba(31,111,235,0.08); border: 1px solid rgba(31,111,235,0.3);
    border-radius: 6px; padding: 10px 14px; margin: 10px 0;
    font-size: 0.8rem; color: #8b949e;
    font-style: italic;
}}

.ms-footer {{
    font-size: 0.7rem; color: {C_TEXT2}; text-align: center;
    margin-top: 40px; padding-top: 14px;
    border-top: 1px solid {C_BORDER};
}}

/* Plotly dark override */
.js-plotly-plot .plotly .bg {{
    fill: {C_BG2} !important;
}}
</style>
""", unsafe_allow_html=True)


# Colour token exports for use in chart helpers
COLORS = {
    "bg":     C_BG,
    "bg2":    C_BG2,
    "bg3":    C_BG3,
    "border": C_BORDER,
    "text":   C_TEXT,
    "text2":  C_TEXT2,
    "accent": C_ACCENT,
    "accept": C_ACCEPT,
    "reject": C_REJECT,
    "review": C_REVIEW,
    "blue":   C_BLUE,
}

DEFECT_COLORS = {
    "Crack":            "#f85149",
    "Surface Scratch":  "#e8500a",
    "Contamination":    "#d29922",
    "Dimensional Error":"#58a6ff",
    "Wrong Label":      "#bc8cff",
    "Missing Part":     "#3fb950",
    "Colour Deviation": "#ffa657",
}

LINE_COLORS = {
    "LINE-A": "#58a6ff",
    "LINE-B": "#3fb950",
    "LINE-C": "#e8500a",
    "LINE-D": "#d29922",
    "LINE-E": "#bc8cff",
}
