import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64

def load_image_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""

logo_base64 = load_image_base64("assets/inverters_logo.png")


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PH Labor Market Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session State ─────────────────────────────────────────────────────────────
if "nav" not in st.session_state:
    st.session_state.nav = "overview"
if "data_mode" not in st.session_state:
    st.session_state.data_mode = "raw"
if "chart_unit" not in st.session_state:
    st.session_state.chart_unit = "share"
if "proj_model" not in st.session_state:
    st.session_state.proj_model = "linear"
if "cluster_filter" not in st.session_state:
    st.session_state.cluster_filter = "High Growth"
if "proj_tab" not in st.session_state:
    st.session_state.proj_tab = "fit"
if "clust_tab" not in st.session_state:
    st.session_state.clust_tab = "scatter"

def update_proj_model():
    if "proj_model_dropdown" in st.session_state:
        st.session_state.proj_model = "linear" if st.session_state.proj_model_dropdown == "Linear" else "quadratic"

def update_cluster_filter():
    if "cluster_inspect_dropdown" in st.session_state:
        st.session_state.cluster_filter = st.session_state.cluster_inspect_dropdown

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300..800;1,300..800&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── App background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #f0f3fa !important;
    font-family: 'Inter', sans-serif !important;
    overflow: hidden !important;
    height: 100vh !important;
}

/* ── Enforce Italics ── */
em, i {
    font-style: italic !important;
}

/* ── Hide Streamlit chrome and remove top offsets ── */
.stAppHeader,
[data-testid="stAppHeader"],
#MainMenu, footer, header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ── Remove Streamlit main container top offset ── */
.stMain,
[data-testid="stMain"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* ── Collapse empty style container blocks ── */
[data-testid="element-container"]:has(style) {
    display: none !important;
}

/* ── Main block ── */
.stMainBlockContainer,
[data-testid="stMainBlockContainer"],
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1.8rem !important;
    padding-left: 1.6rem !important;
    padding-right: 1.6rem !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
    height: 100vh !important;
    overflow: hidden !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #162040 !important;
    border-right: none !important;
    min-width: 210px !important;
    max-width: 210px !important;
    overflow: visible !important;
    z-index: 99999 !important;
    position: relative !important;
    height: 100vh !important;
}
[data-testid="stSidebarContent"] {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: visible !important;
    padding-bottom: 75px !important;
}
[data-testid="stSidebarHeader"],
.stSidebarHeader {
    display: none !important;
    background-color: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebarUserContent"],
[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 0 !important;
    padding-top: 0 !important;
    margin-top: 0 !important;
    overflow: visible !important;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.1rem !important;
}

/* Sidebar widgets padding */
[data-testid="stSidebar"] .stRadio,
[data-testid="stSidebar"] .stDownloadButton {
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
}
[data-testid="stSidebar"] .stRadio {
    margin-bottom: 0.2rem !important;
}

/* Radio options styling to prevent truncation and make them look clean */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    font-size: 0.8rem !important;
    color: #cbd5e1 !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
    margin-right: 0.5rem !important;
    margin-left: 0 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    gap: 0.3rem !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #93c5fd !important;
}

/* Sidebar section label */
.sb-section-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #93c5fd;
    padding: 0 0.8rem;
    margin-bottom: 0.1rem;
    margin-top: 0.6rem;
    display: block;
}

/* Sidebar hover-tooltip rows */
.sb-row {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.22rem 0.8rem;
    cursor: default;
    transition: background 0.12s;
    z-index: 1;
}
.sb-row:hover {
    background: rgba(255,255,255,0.05);
    z-index: 100;
}
.sb-row-left {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: #e2e8f0;
    font-weight: 500;
}
.sb-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sb-info-icon {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.3);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    cursor: default;
    position: relative;
}
/* Tooltip appears to the RIGHT of the info icon */
.sb-tooltip {
    display: none;
    position: absolute;
    top: 50%;
    left: calc(100% + 10px);
    transform: translateY(-50%);
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 7px;
    padding: 0.5rem 0.7rem;
    font-size: 0.71rem;
    color: #cbd5e1;
    line-height: 1.45;
    z-index: 9999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    pointer-events: none;
    font-family: 'Inter', sans-serif;
    font-style: normal;
    font-weight: 400;
    white-space: normal;
    width: 200px;
}
.sb-row:hover .sb-tooltip {
    display: block;
}

/* Sidebar divider */
.sb-divider {
    height: 1px;
    background: #1e2e55;
    margin: 0.7rem 0.8rem 0;
}

/* Fixed export button at bottom of sidebar */
[data-testid="stSidebar"] .stDownloadButton {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    width: 210px !important;
    background: #162040 !important;
    padding: 0.75rem 0.8rem 1.1rem 0.8rem !important;
    border-top: 1px solid #1e2e55 !important;
    z-index: 999999 !important;
}

/* Sidebar download button */
[data-testid="stSidebar"] .stDownloadButton button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    padding: 0.35rem 0.6rem !important;
    width: 100% !important;
    text-align: center !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    letter-spacing: 0.2px !important;
    margin-top: 0.1rem !important;
}
[data-testid="stSidebar"] .stDownloadButton button:hover {
    background: rgba(255,255,255,0.13) !important;
    border-color: rgba(255,255,255,0.3) !important;
    color: #ffffff !important;
}

/* ── Nav buttons in sidebar ── */
[data-testid="stSidebar"] .stButton button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #7a8db0 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.65rem 0.8rem !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
}

/* Sidebar radio (filters) */
[data-testid="stSidebar"] .stRadio > label {
    color: #93c5fd !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    text-align: left !important;
    display: block !important;
    width: 100% !important;
    margin-bottom: 0.25rem !important;
    margin-top: 0.2rem !important;
}
[data-testid="stSidebar"] .stRadio > label div p,
[data-testid="stSidebar"] .stSelectbox > label div p {
    text-align: left !important;
}
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.1rem !important;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    text-align: left !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 0.1rem 0 !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stSelectbox > label {
    color: #93c5fd !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    text-align: left !important;
    display: block !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1e2e55 !important;
    border: 1px solid #2a3f6f !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox div[role="button"],
[data-testid="stSidebar"] .stSelectbox div[role="button"] span,
[data-testid="stSidebar"] .stSelectbox [data-testid="stSelectboxInput"] span,
[data-testid="stSidebar"] .stSelectbox p {
    color: #ffffff !important;
    font-size: 0.88rem !important;
    text-align: left !important;
    width: 100% !important;
}

/* ── Disable typing in all selectboxes ── */
.stSelectbox input {
    pointer-events: none !important;
    caret-color: transparent !important;
    user-select: none !important;
}
.stSelectbox,
.stSelectbox *,
.stSelectbox > div,
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select"] * {
    cursor: default !important;
}
.stSelectbox [data-baseweb="select"] > div:focus,
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-visible,
.stSelectbox [data-baseweb="select"] > div[aria-expanded="true"],
.stSelectbox [data-baseweb="select"] > div {
    border-color: rgba(255,255,255,0.15) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 0.6rem 0.9rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    position: relative;
    height: 100%;
}
.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.3rem;
}
.kpi-label {
    font-size: 0.70rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
}
.kpi-icon {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    flex-shrink: 0;
}
.kpi-value {
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1;
    color: #1e293b;
    margin-bottom: 0.1rem;
}
.kpi-delta {
    font-size: 0.68rem;
    color: #94a3b8;
}
.kpi-delta .up   { color: #10b981; font-weight: 600; }
.kpi-delta .down { color: #ef4444; font-weight: 600; }

/* ── Chart card ── */
.chart-card,
.st-key-card_dynamics,
.st-key-card_breakdown,
.st-key-card_projections,
.st-key-card_fit,
.st-key-card_yoy,
.st-key-card_kmeans,
.st-key-card_cagr,
.st-key-card_detail,
.st-key-card_summary_chart {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.2rem 0.4rem 1.2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important;
    border: none !important;
    height: 100% !important;
    margin-top: -0.8rem !important;
}
.st-key-card_summary_text {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.2rem 0.7rem 1.2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important;
    border: none !important;
    height: auto !important;
    margin-top: -0.8rem !important;
}
.st-key-card_tier_table {
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 0.85rem 1.2rem 0.9rem 1.2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important;
    border: none !important;
    height: auto !important;
    margin-top: -0.8rem !important;
}
.chart-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.15rem;
}
.chart-card-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-bottom: 0.5rem;
}

/* ── Streamlit Tabs Styling ── */
div[data-testid="stTabBar"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    padding: 0.2rem 0.25rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important;
    border: 1px solid #dde3ef !important;
    margin-top: -0.8rem !important;
    margin-bottom: 0.55rem !important;
    height: 34px !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
}
div[data-testid="stTabBar"] button[data-testid="stTab"] {
    background-color: transparent !important;
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    padding: 0.2rem 0.6rem !important;
    border-radius: 6px !important;
    border: none !important;
    transition: all 0.15s !important;
    margin: 0 0.1rem !important;
    height: 26px !important;
    min-height: 26px !important;
    line-height: 1.2 !important;
}
div[data-testid="stTabBar"] button[data-testid="stTab"]:hover {
    color: #1e293b !important;
    background-color: #f1f5f9 !important;
}
div[data-testid="stTabBar"] button[data-testid="stTab"][aria-selected="true"] {
    background-color: #162040 !important;
    color: #ffffff !important;
}
/* Hide default BaseWeb bottom line and highlight */
div[data-testid="stTabBar"] div[data-baseweb="tab-highlight"],
div[data-testid="stTabBar"]::after {
    display: none !important;
    height: 0 !important;
}

/* ── Page title ── */
.page-title {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #162040 !important;
    text-align: center !important;
    margin-top: -0.5rem !important;
    margin-bottom: 0.9rem !important;
    width: 100% !important;
    line-height: 1.25 !important;
}
.page-sub {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: 0.1rem;
    margin-bottom: 0.5rem;
}

/* ── Finding pill ── */
.finding-bar {
    background: #ffffff;
    border-radius: 10px;
    padding: 0.5rem 1.0rem;
    border-left: 3px solid #162040;
    box-shadow: 0 1px 5px rgba(0,0,0,0.05);
    font-size: 0.76rem;
    color: #475569;
    line-height: 1.5;
    margin-bottom: 0.6rem;
}
.finding-bar strong { color: #162040; }

/* ── Cluster badge ── */
.cluster-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.2rem 0.75rem;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    border: 1px solid;
}

/* ── Occupation card ── */
.occ-card {
    background: #f8fafc;
    border: 1px solid #e8edf5;
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    margin-bottom: 0.45rem;
}
.occ-name {
    font-size: 0.76rem;
    font-weight: 600;
    color: #1e293b;
}
.occ-row2 {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.2rem;
}
.occ-cagr {
    font-size: 0.76rem;
    font-weight: 700;
}
.occ-meta {
    font-size: 0.67rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}
.occ-bar-bg {
    background: #e8edf5;
    border-radius: 999px;
    height: 4px;
    margin-top: 0.3rem;
    overflow: hidden;
}

/* ── R2 bars ── */
.r2-row { margin-bottom: 0.85rem; }
.r2-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.74rem;
    color: #475569;
    margin-bottom: 0.3rem;
    font-weight: 600;
}
.r2-bg {
    background: #f1f5f9;
    border-radius: 999px;
    height: 7px;
    overflow: hidden;
}

/* ── Proj table rows ── */
.proj-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.78rem;
}

/* hide Streamlit element labels inside sidebar buttons */
[data-testid="stSidebar"] .stButton > div { padding: 0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Color palette ─────────────────────────────────────────────────────────────
C_HIGH = "#162040"   # navy   — High Skill accent (matches sidebar)
C_MID  = "#f5a623"   # amber  — Middle Skill
C_LOW  = "#3b82f6"   # blue   — Low Skill
C_PINK = "#ec4899"   # pink   — sub-series
NAVY   = "#162040"

def hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def plotly_theme(h=300, ml=40, mr=16, mt=8, mb=8):
    return dict(
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8", size=11),
        margin=dict(l=ml, r=mr, t=mt, b=mb),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05,
            xanchor="right", x=1,
            font=dict(size=10, color="#64748b"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color="#94a3b8"),
            linecolor="#e2e8f0",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#f1f5f9",
            zeroline=False,
            tickfont=dict(size=10, color="#94a3b8"),
        ),
    )

# ─── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("1B3GEMP4.csv", sep=";", skiprows=2)
    df = df.replace(".", np.nan)
    for col in df.columns[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"])
    df = df.dropna(subset=["Total"]).sort_values("Year")
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

def compute_adjusted(df):
    df_adj = df.copy()
    total_2018 = df_adj.loc[df_adj["Year"] == 2018, "Total"].values[0]
    m2018      = df_adj.loc[df_adj["Year"] == 2018, "Managers"].values[0]
    share_2018 = m2018 / total_2018
    for idx, row in df_adj.iterrows():
        if row["Year"] >= 2019:
            adj  = row["Total"] * share_2018
            diff = adj - row["Managers"]
            ss, cs = row["Service and Sales Workers"], row["Clerical Support Workers"]
            s = ss + cs
            if s > 0:
                df_adj.at[idx, "Managers"] = adj
                df_adj.at[idx, "Service and Sales Workers"]  = max(0, ss - diff * (ss / s))
                df_adj.at[idx, "Clerical Support Workers"]   = max(0, cs - diff * (cs / s))
    return df_adj

df_adjusted = compute_adjusted(df_raw)

high_cols   = ["Managers", "Professionals", "Technicians and Associate Professionals"]
middle_cols = ["Clerical Support Workers", "Craft and Related Trades Workers", "Plant and Machine Operators and Assemblers"]
low_cols    = ["Service and Sales Workers", "Skilled Agricultural Forestry and Fishery Workers", "Elementary Occupations"]
occ_groups  = high_cols + middle_cols + low_cols + ["Armed Forces Occupations"]

def add_tiers(df):
    d = df.copy()
    d["High Skill"]   = d[high_cols].sum(axis=1)
    d["Middle Skill"] = d[middle_cols].sum(axis=1)
    d["Low Skill"]    = d[low_cols].sum(axis=1)
    d["High Share"]   = d["High Skill"]   / d["Total"] * 100
    d["Middle Share"] = d["Middle Skill"] / d["Total"] * 100
    d["Low Share"]    = d["Low Skill"]    / d["Total"] * 100
    return d

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + Brand
    st.markdown(f"""
    <div style="padding:1.2rem 0.8rem 1.0rem 0.8rem; display:flex; justify-content:center;">
      <div style="display:flex;align-items:center;gap:1.0rem;margin-bottom:0.4rem;">
        <div style="width:42px;height:42px;border-radius:50%;background:#ffffff;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;overflow:hidden;">
          <img src="data:image/png;base64,{logo_base64}" style="width:100%;height:100%;object-fit:contain;padding:4px;" />
        </div>
        <div>
          <div style="font-size:1.1rem;font-weight:700;color:#ffffff;line-height:1.2;">Inverters</div>
          <div style="font-size:0.72rem;color:#cbd5e1;white-space:nowrap;">CMSC 178DA</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0 0 4.0rem 0; margin-top: 0.1rem;">
      <div style="height:1px;background:#1e2e55;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    data_mode_opt = st.radio(
        "Data Model",
        ["Raw PSA Figures", "Adjusted (Smoothed)"],
        index=0 if st.session_state.data_mode == "raw" else 1,
    )
    st.session_state.data_mode = "raw" if "Raw" in data_mode_opt else "adj"


# ─── Active dataset & KPI variables ────────────────────────────────────────────
adf = add_tiers(df_raw if st.session_state.data_mode == "raw" else df_adjusted)

h16 = adf.loc[adf["Year"] == 2016, "High Share"].values[0]
h25 = adf.loc[adf["Year"] == 2025, "High Share"].values[0]
m16 = adf.loc[adf["Year"] == 2016, "Middle Share"].values[0]
m25 = adf.loc[adf["Year"] == 2025, "Middle Share"].values[0]
l16 = adf.loc[adf["Year"] == 2016, "Low Share"].values[0]
l25 = adf.loc[adf["Year"] == 2025, "Low Share"].values[0]
total_16 = adf.loc[adf["Year"] == 2016, "Total"].values[0]
total_25 = adf.loc[adf["Year"] == 2025, "Total"].values[0]
dh = h25 - h16
dm = m25 - m16
dl = l25 - l16
dtot = (total_25 - total_16) / 1000
nav = st.session_state.nav

# ─── Sidebar Info & Export ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sb-divider"></div>
    <span class="sb-section-label">Skill Tier Classification</span>
    <div class="sb-row">
      <div class="sb-row-left">
        <div class="sb-dot" style="background:#ffffff;"></div>
        <span>High Skill</span>
      </div>
      <div class="sb-info-icon">
        <span style="font-family:Georgia,serif;font-style:italic;font-size:0.72rem;color:#94a3b8;line-height:1;user-select:none;">i</span>
        <div class="sb-tooltip"><strong>High Skill</strong><br>Managers, Professionals, Technicians &amp; Associate Professionals</div>
      </div>
    </div>
    <div class="sb-row">
      <div class="sb-row-left">
        <div class="sb-dot" style="background:{C_MID};"></div>
        <span>Middle Skill</span>
      </div>
      <div class="sb-info-icon">
        <span style="font-family:Georgia,serif;font-style:italic;font-size:0.72rem;color:#94a3b8;line-height:1;user-select:none;">i</span>
        <div class="sb-tooltip"><strong>Middle Skill</strong><br>Clerical Support Workers, Craft &amp; Related Trades, Plant &amp; Machine Operators</div>
      </div>
    </div>
    <div class="sb-row">
      <div class="sb-row-left">
        <div class="sb-dot" style="background:{C_LOW};"></div>
        <span>Low Skill</span>
      </div>
      <div class="sb-info-icon">
        <span style="font-family:Georgia,serif;font-style:italic;font-size:0.72rem;color:#94a3b8;line-height:1;user-select:none;">i</span>
        <div class="sb-tooltip"><strong>Low Skill</strong><br>Service &amp; Sales Workers, Agricultural &amp; Fishery Workers, Elementary Occupations</div>
      </div>
    </div>

    <div class="sb-divider"></div>
    <span class="sb-section-label">Dataset</span>
    <div class="sb-row">
      <div class="sb-row-left"><span>Source</span></div>
      <div class="sb-info-icon">
        <span style="font-family:Georgia,serif;font-style:italic;font-size:0.72rem;color:#94a3b8;line-height:1;user-select:none;">i</span>
        <div class="sb-tooltip">Philippine Statistics Authority OpenSTAT — Labor Force Survey (LFS)</div>
      </div>
    </div>
    <div class="sb-divider"></div>
    """, unsafe_allow_html=True)

    st.download_button(
        label="Export Active Dataset (CSV)",
        data=adf.to_csv(index=False),
        file_name=f"ph_labor_market_{st.session_state.data_mode}.csv",
        mime="text/csv",
        use_container_width=True
    )



# ─── Persistent Title Question ────────────────────────────────────────────────
st.markdown("""
<div class="page-title">Has the Philippine labor market polarized into an hourglass structure from 2016 to 2025?</div>
""", unsafe_allow_html=True)

# ─── Persistent KPI Cards Row ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4, gap="small")
with k1:
    arrow = "up" if dh >= 0 else "down"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-top">
        <div class="kpi-label">High-Skill Share</div>
        <div class="kpi-icon" style="background:#f0f3fa;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{C_HIGH}" stroke-width="2.2">
            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>
          </svg>
        </div>
      </div>
      <div class="kpi-value" style="color:{C_HIGH};">{h25:.1f}%</div>
      <div class="kpi-delta">Since 2016: <span class="{arrow}">{dh:+.1f} pp</span></div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    arrow = "up" if dm >= 0 else "down"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-top">
        <div class="kpi-label">Middle-Skill Share</div>
        <div class="kpi-icon" style="background:#fff8ed;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{C_MID}" stroke-width="2.2">
            <line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>
          </svg>
        </div>
      </div>
      <div class="kpi-value" style="color:{C_MID};">{m25:.1f}%</div>
      <div class="kpi-delta">Since 2016: <span class="{arrow}">{dm:+.1f} pp</span></div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    arrow = "up" if dl >= 0 else "down"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-top">
        <div class="kpi-label">Low-Skill Share</div>
        <div class="kpi-icon" style="background:#eff6ff;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="{C_LOW}" stroke-width="2.2">
            <circle cx="12" cy="12" r="10"/><polyline points="12 8 12 12 14 14"/>
          </svg>
        </div>
      </div>
      <div class="kpi-value" style="color:{C_LOW};">{l25:.1f}%</div>
      <div class="kpi-delta">Since 2016: <span class="{arrow}">{dl:+.1f} pp</span></div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    arrow_tot = "up" if dtot >= 0 else "down"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-top">
        <div class="kpi-label">Total Employed · 2025</div>
        <div class="kpi-icon" style="background:#f0f3fa;">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2.2">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
          </svg>
        </div>
      </div>
      <div class="kpi-value" style="color:{C_HIGH};">{total_25/1000:.2f}M</div>
      <div class="kpi-delta">Since 2016: <span class="{arrow_tot}">{dtot:+.2f}M</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 0.7rem;'></div>", unsafe_allow_html=True)

# ─── Navigation Tabs (Compact & Full Width) ────────────────────────────────────
t1, t2, t3, t4 = st.columns(4, gap="small")
with t1:
    if st.button("Overview", key="tab_overview", use_container_width=True):
        st.session_state.nav = "overview"
        st.rerun()
with t2:
    if st.button("Skill Tier Trends", key="tab_trends", use_container_width=True):
        st.session_state.nav = "trends"
        st.rerun()
with t3:
    if st.button("Trend Projections", key="tab_proj", use_container_width=True):
        st.session_state.nav = "proj"
        st.rerun()
with t4:
    if st.button("Occupation Clustering", key="tab_clust", use_container_width=True):
        st.session_state.nav = "clust"
        st.rerun()

# Style the active navigation button compact CSS injection
active_tab = st.session_state.nav
st.markdown(f"""
<style>
.st-key-tab_overview button,
.st-key-tab_trends button,
.st-key-tab_proj button,
.st-key-tab_clust button {{
    background: #ffffff !important;
    border: 1px solid #dde3ef !important;
    color: #64748b !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    padding: 0.25rem 0.5rem !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    height: 28px !important;
    min-height: 28px !important;
    line-height: 1.2 !important;
}}
.st-key-tab_{active_tab} button {{
    background: #162040 !important;
    color: #ffffff !important;
    border-color: #162040 !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── Render Overview View ───
if nav == "overview":
    # Finding calculation
    if st.session_state.data_mode == "raw":
        finding = "<strong>No clear hourglass found.</strong> Middle-skill roles expanded (+4.8 pp) alongside low-skill growth — driven partly by an administrative Managers reclassification. Toggle to <em>Adjusted</em> for the economic signal."
    else:
        finding = "<strong>Partial polarization detected.</strong> High-skill share stayed stable; low-skill services expanded fastest (+3.6 pp); middle-skill grew modestly (+2.1 pp). The hourglass pattern is present but not pronounced."

    # Dynamic details based on raw vs adjusted data model
    if st.session_state.data_mode == "raw":
        verdict_title = "NO HOURGLASS STRUCTURE FOUND"
        verdict_color = "#3b82f6"  # Blue
        verdict_bg = "#eff6ff"
        verdict_desc = "Standard figures show middle-skill growth (+4.8 pp) driven by LFS definition shifts. Toggle to 'Adjusted' to see the corrected economic signal."
        high_status = f"{dh:+.1f} pp · Administrative Drop"
        high_desc = "Managers fell due to LFS reclassification."
        mid_status = f"{dm:+.1f} pp · Artificially Inflated"
        mid_desc = "Clerical workers gained share from reclassification."
        low_status = f"{dl:+.1f} pp · Moderate Growth"
        low_desc = "Low-skill sales and service roles expanded."
    else:
        verdict_title = "PARTIAL POLARIZATION DETECTED"
        verdict_color = "#f5a623"  # Amber
        verdict_bg = "#fff8ed"
        verdict_desc = "Low-skill roles expanded fastest (+3.6 pp) while high-skill stayed flat. Middle-skill grew modestly (+2.1 pp) instead of hollowing out."
        high_status = f"{dh:+.1f} pp · Stable"
        high_desc = "High-skill share held steady around 13-14%."
        mid_status = f"{dm:+.1f} pp · Resilient"
        mid_desc = "Grew modestly; no structural collapse detected."
        low_status = f"{dl:+.1f} pp · Growth Driver"
        low_desc = "Low-skill service sectors expanded the fastest."

    # Summary container (Full Width Card Design)
    with st.container(key="card_summary_text"):
        st.markdown(f"""
        <div class="chart-card-title">Executive Summary</div>
        <div style="margin-top:0.5rem; margin-bottom:0.5rem;">
          <!-- Verdict Card -->
          <div style="background:{verdict_bg}; border-left: 4px solid {verdict_color}; padding: 0.6rem 0.8rem; border-radius: 8px;">
            <div style="font-size: 0.65rem; font-weight: 700; color: {verdict_color}; text-transform: uppercase; letter-spacing: 0.5px;">Key Verdict</div>
            <div style="font-size: 1.0rem; font-weight: 800; color: #1e293b; margin: 0.1rem 0; line-height: 1.2;">{verdict_title}</div>
            <div style="font-size: 0.74rem; color: #475569; line-height: 1.35;">{verdict_desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Three Tier Cards in native columns for consistent alignment and margins
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.5rem 0.7rem;">
              <div style="font-size: 0.62rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">High-Skill Tier</div>
              <div style="font-size: 0.8rem; font-weight: 700; color: {C_HIGH}; margin: 0.05rem 0;">{high_status}</div>
              <div style="font-size: 0.7rem; color: #475569; line-height: 1.25;">{high_desc}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.5rem 0.7rem;">
              <div style="font-size: 0.62rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Middle-Skill Tier</div>
              <div style="font-size: 0.8rem; font-weight: 700; color: {C_MID}; margin: 0.05rem 0;">{mid_status}</div>
              <div style="font-size: 0.7rem; color: #475569; line-height: 1.25;">{mid_desc}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.5rem 0.7rem;">
              <div style="font-size: 0.62rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Low-Skill Tier</div>
              <div style="font-size: 0.8rem; font-weight: 700; color: {C_LOW}; margin: 0.05rem 0;">{low_status}</div>
              <div style="font-size: 0.7rem; color: #475569; line-height: 1.25;">{low_desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Skill Tier Snapshot table (merged inside this card) ──
        high_abs_16 = adf.loc[adf["Year"] == 2016, "High Skill"].values[0]
        high_abs_25 = adf.loc[adf["Year"] == 2025, "High Skill"].values[0]
        mid_abs_16  = adf.loc[adf["Year"] == 2016, "Middle Skill"].values[0]
        mid_abs_25  = adf.loc[adf["Year"] == 2025, "Middle Skill"].values[0]
        low_abs_16  = adf.loc[adf["Year"] == 2016, "Low Skill"].values[0]
        low_abs_25  = adf.loc[adf["Year"] == 2025, "Low Skill"].values[0]

        def trend_arrow(delta):
            return "↑" if delta > 0 else "↓"

        def delta_color(delta, tier):
            if tier == "high":
                return C_HIGH if delta > 0 else "#ef4444"
            elif tier == "mid":
                return "#10b981" if delta > 0 else "#ef4444"
            else:
                return C_LOW if delta > 0 else "#ef4444"

        st.markdown(f"""
        <div style="margin-top: 0.75rem; border-top: 1px solid #e2e8f0; padding-top: 0.6rem;">
          <div class="chart-card-title" style="margin-bottom: 0.4rem;">Skill Tier Snapshot</div>
          <div style="overflow-x: auto;">
            <table style="width:100%; border-collapse: collapse; font-size: 0.72rem;">
              <thead>
                <tr style="border-bottom: 2px solid #e2e8f0;">
                  <th style="text-align:left; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Skill Tier</th>
                  <th style="text-align:right; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">2016 Share</th>
                  <th style="text-align:right; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">2025 Share</th>
                  <th style="text-align:right; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Change (pp)</th>
                  <th style="text-align:right; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">2016 Jobs (K)</th>
                  <th style="text-align:right; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">2025 Jobs (K)</th>
                  <th style="text-align:center; padding: 0.28rem 0.5rem; color:#94a3b8; font-size:0.63rem; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Trend</th>
                </tr>
              </thead>
              <tbody>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                  <td style="padding: 0.3rem 0.5rem; font-weight:700; color:{C_HIGH};">High Skill</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{h16:.1f}%</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{h25:.1f}%</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; font-weight:700; color:{delta_color(dh, 'high')};">{trend_arrow(dh)} {dh:+.1f}</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{high_abs_16:,.0f}</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{high_abs_25:,.0f}</td>
                  <td style="text-align:center; padding: 0.3rem 0.5rem; font-size:0.9rem; color:{delta_color(dh, 'high')};">{trend_arrow(dh)}</td>
                </tr>
                <tr style="border-bottom: 1px solid #f1f5f9;">
                  <td style="padding: 0.3rem 0.5rem; font-weight:700; color:{C_MID};">Middle Skill</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{m16:.1f}%</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{m25:.1f}%</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; font-weight:700; color:{delta_color(dm, 'mid')};">{trend_arrow(dm)} {dm:+.1f}</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{mid_abs_16:,.0f}</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{mid_abs_25:,.0f}</td>
                  <td style="text-align:center; padding: 0.3rem 0.5rem; font-size:0.9rem; color:{delta_color(dm, 'mid')};">{trend_arrow(dm)}</td>
                </tr>
                <tr>
                  <td style="padding: 0.3rem 0.5rem; font-weight:700; color:{C_LOW};">Low Skill</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{l16:.1f}%</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{l25:.1f}%</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; font-weight:700; color:{delta_color(dl, 'low')};">{trend_arrow(dl)} {dl:+.1f}</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{low_abs_16:,.0f}</td>
                  <td style="text-align:right; padding: 0.3rem 0.5rem; color:#475569;">{low_abs_25:,.0f}</td>
                  <td style="text-align:center; padding: 0.3rem 0.5rem; font-size:0.9rem; color:{delta_color(dl, 'low')};">{trend_arrow(dl)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Render Trends View ───
elif nav == "trends":
    col_l, col_r = st.columns([3, 2], gap="small")

    with col_l:
        with st.container(key="card_dynamics"):
            hdr_l, hdr_r = st.columns([3, 1])
            with hdr_l:
                st.markdown("""
                <div class="chart-card-title">Employment Share Dynamics</div>
                <div class="chart-card-sub">High / Middle / Low Skill  ·  2016 – 2025</div>
                """, unsafe_allow_html=True)
            with hdr_r:
                chart_unit_opt = st.selectbox(
                    "Units",
                    ["% Share", "Absolute (K)"],
                    index=0 if st.session_state.chart_unit == "share" else 1,
                    key="chart_unit_dropdown",
                    label_visibility="collapsed",
                )
                st.session_state.chart_unit = "share" if "Share" in chart_unit_opt else "abs"

            if st.session_state.chart_unit == "share":
                fig = go.Figure()
                for label, col, color in [
                    ("High Skill",   "High Share",   C_HIGH),
                    ("Middle Skill", "Middle Share",  C_MID),
                    ("Low Skill",    "Low Share",     C_LOW),
                ]:
                    fig.add_trace(go.Scatter(
                        x=adf["Year"], y=adf[col], name=label,
                        mode="lines+markers",
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, color=color, line=dict(width=2, color="#ffffff")),
                        fill="tozeroy",
                        fillcolor=hex_rgba(color, 0.08),
                    ))
                lay = plotly_theme(h=315)
                lay["yaxis"]["title"] = "Share (%)"
                fig.update_layout(**lay)
            else:
                fig = go.Figure()
                for label, col, color in [
                    ("High Skill",   "High Skill",   C_HIGH),
                    ("Middle Skill", "Middle Skill",  C_MID),
                    ("Low Skill",    "Low Skill",     C_LOW),
                ]:
                    fig.add_trace(go.Scatter(
                        x=adf["Year"], y=adf[col], name=label,
                        mode="lines+markers",
                        line=dict(color=color, width=2.5),
                        marker=dict(size=7, color=color, line=dict(width=2, color="#ffffff")),
                        stackgroup="one",
                        fillcolor=hex_rgba(color, 0.75),
                    ))
                lay = plotly_theme(h=315)
                lay["yaxis"]["title"] = "Employment (Thousands)"
                fig.update_layout(**lay)

            st.plotly_chart(fig, width='stretch', config=dict(displayModeBar=False))

    with col_r:
        with st.container(key="card_breakdown"):
            st.markdown("""
            <div class="chart-card-title">Middle-Skill Breakdown</div>
            <div class="chart-card-sub">Clerical  ·  Craft & Trades  ·  Plant Operators</div>
            """, unsafe_allow_html=True)

            short_names = {
                "Clerical Support Workers":                   "Clerical",
                "Craft and Related Trades Workers":            "Craft & Trades",
                "Plant and Machine Operators and Assemblers":  "Plant Operators",
            }
            bar_colors = ["#d97706", "#f5a623", "#fbbf24"]  # shades of Amber/Orange (middle-skill theme)
            fig2 = go.Figure()
            for (col, name), color in zip(short_names.items(), bar_colors):
                fig2.add_trace(go.Bar(
                    x=adf["Year"], y=adf[col], name=name,
                    marker=dict(color=color, opacity=0.88, line=dict(width=0)),
                ))
            lay2 = plotly_theme(h=315)
            lay2["barmode"] = "group"
            lay2["bargap"]  = 0.22
            lay2["yaxis"]["title"] = "Thousands"
            fig2.update_layout(**lay2)
            st.plotly_chart(fig2, width='stretch', config=dict(displayModeBar=False))

    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "proj":
    degree = 1 if st.session_state.proj_model == "linear" else 2
    years  = adf["Year"].values
    future = np.arange(2016, 2031)

    proj_shares, r2_scores = {}, {}
    for label, col in [
        ("High Skill",   "High Share"),
        ("Middle Skill", "Middle Share"),
        ("Low Skill",    "Low Share"),
    ]:
        vals      = adf[col].values
        coefs     = np.polyfit(years, vals, degree)
        poly      = np.poly1d(coefs)
        fitted    = poly(years)
        projected = np.clip(poly(future), 0, 100)
        proj_shares[label] = projected
        ss_res = np.sum((vals - fitted) ** 2)
        ss_tot = np.sum((vals - vals.mean()) ** 2)
        r2_scores[label] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    col_l, col_r = st.columns([3, 2], gap="small")

    with col_l:
        with st.container(key="card_projections"):
            hdr_l, hdr_r = st.columns([3, 1])
            with hdr_r:
                st.selectbox(
                    "Projection Model",
                    ["Linear", "Quadratic"],
                    index=0 if st.session_state.proj_model == "linear" else 1,
                    key="proj_model_dropdown",
                    on_change=update_proj_model,
                    label_visibility="collapsed",
                )
            model_label = "Linear" if st.session_state.proj_model == "linear" else "Quadratic"
            with hdr_l:
                st.markdown(f"""
                <div class="chart-card-title">Employment Share Projections — 2030 Outlook</div>
                <div class="chart-card-sub">{model_label} regression  ·  Dashed = Forecast  ·  Dots = Actual data</div>
                """, unsafe_allow_html=True)

            colors_p = {"High Skill": C_HIGH, "Middle Skill": C_MID, "Low Skill": C_LOW}
            fig_p = go.Figure()
            fig_p.add_vline(
                x=2025, line_dash="dot", line_color="#cbd5e1", line_width=1.5,
                annotation_text="2025 (Latest)", annotation_position="top",
                annotation_font_size=9, annotation_font_color="#94a3b8",
            )
            for label, col in [
                ("High Skill",   "High Share"),
                ("Middle Skill", "Middle Share"),
                ("Low Skill",    "Low Share"),
            ]:
                clr = colors_p[label]
                fig_p.add_trace(go.Scatter(
                    x=adf["Year"], y=adf[col], name=label,
                    mode="markers",
                    marker=dict(size=9, color=clr, symbol="circle",
                                line=dict(width=2, color="#ffffff")),
                ))
                fig_p.add_trace(go.Scatter(
                    x=future, y=proj_shares[label],
                    mode="lines",
                    line=dict(color=clr, width=2.5, dash="dot"),
                    showlegend=False,
                ))
            lay_p = plotly_theme(h=308, mb=12)
            lay_p["xaxis"]["range"] = [2015, 2031]
            lay_p["yaxis"]["title"] = "Share (%)"
            fig_p.update_layout(**lay_p)
            st.plotly_chart(fig_p, width='stretch', config=dict(displayModeBar=False))

    with col_r:
        # Mini tab buttons styled like the main tabs but red-orange (#ff4b4b)
        p1, p2 = st.columns(2, gap="small")
        with p1:
            if st.button("Model Fit & Forecast", key="btn_fit", use_container_width=True):
                st.session_state.proj_tab = "fit"
                st.rerun()
        with p2:
            if st.button("YoY Share Change", key="btn_yoy", use_container_width=True):
                st.session_state.proj_tab = "yoy"
                st.rerun()

        active_proj_tab = st.session_state.proj_tab
        st.markdown(f"""
        <style>
        /* Shift only the inner buttons row up to align with left card top */
        [data-testid="stColumn"] [data-testid="stHorizontalBlock"]:has(.st-key-btn_fit) {{
            margin-top: -0.8rem !important;
            margin-bottom: 0 !important;
        }}
        /* Button styling */
        .st-key-btn_fit button,
        .st-key-btn_yoy button {{
            background: #ffffff !important;
            border: 1px solid #dde3ef !important;
            color: #64748b !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.72rem !important;
            padding: 0.25rem 0.5rem !important;
            transition: all 0.15s !important;
            box-shadow: none !important;
            height: 28px !important;
            min-height: 28px !important;
            line-height: 1.2 !important;
        }}
        .st-key-btn_{active_proj_tab} button {{
            background: #162040 !important;
            color: #ffffff !important;
            border-color: #162040 !important;
        }}
        
        /* Align card bottoms by making columns stretch using flexbox */
        div[data-testid="stColumn"]:has(.st-key-card_projections) > div,
        div[data-testid="stColumn"]:has(.st-key-card_fit) > div,
        div[data-testid="stColumn"]:has(.st-key-card_yoy) > div {{
            height: 100% !important;
        }}

        div[data-testid="stColumn"]:has(.st-key-card_projections) [data-testid="stVerticalBlock"],
        div[data-testid="stColumn"]:has(.st-key-card_fit) [data-testid="stVerticalBlock"],
        div[data-testid="stColumn"]:has(.st-key-card_yoy) [data-testid="stVerticalBlock"] {{
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        div[data-testid="stColumn"]:has(.st-key-card_projections) [data-testid="stVerticalBlock"] > div:has(.st-key-card_projections) {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        div[data-testid="stColumn"]:has(.st-key-card_fit) [data-testid="stVerticalBlock"] > div:has(.st-key-card_fit) {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        div[data-testid="stColumn"]:has(.st-key-card_yoy) [data-testid="stVerticalBlock"] > div:has(.st-key-card_yoy) {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        .st-key-card_projections,
        .st-key-card_fit,
        .st-key-card_yoy {{
            flex-grow: 1 !important;
            height: 100% !important;
        }}
        .st-key-card_projections {{
            margin-top: -0.8rem !important;
            padding: 0.95rem 1.2rem 0.95rem 1.2rem !important;
        }}
        .st-key-card_fit,
        .st-key-card_yoy {{
            margin-top: -0.65rem !important;
        }}
        
        .st-key-card_fit {{
            padding: 0.9rem 1.2rem 0.9rem 1.2rem !important;
        }}
        .r2-row {{
            margin-bottom: 0.65rem !important;
        }}
        .proj-row {{
            padding: 0.37rem 0 !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        if active_proj_tab == "fit":
            with st.container(key="card_fit"):
                st.markdown("""
                <div class="chart-card-title">Model Fit — R² Scores</div>
                <div class="chart-card-sub">Goodness-of-fit per skill tier</div>
                """, unsafe_allow_html=True)

                for label, clr in [
                    ("High Skill",   C_HIGH),
                    ("Middle Skill", C_MID),
                    ("Low Skill",    "Low Share" if "Low Share" in locals() else C_LOW),
                ]:
                    # Make sure label is correctly mapped to C_LOW/C_MID/C_HIGH
                    if label == "Low Skill":
                        clr = C_LOW
                    r2  = r2_scores[label]
                    pct = int(r2 * 100)
                    strength = "Strong" if r2 > 0.8 else "Moderate" if r2 > 0.5 else "Weak"
                    st.markdown(f"""
                    <div class="r2-row">
                      <div class="r2-label-row">
                        <span>{label}</span>
                        <span>R² = <strong style="color:{clr};">{r2:.4f}</strong>
                          &nbsp;<span style="color:#94a3b8;font-weight:400;">· {strength}</span>
                        </span>
                      </div>
                      <div class="r2-bg">
                        <div style="width:{pct}%;height:100%;border-radius:999px;background:{clr};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("""
                <div style="margin-top:0.55rem;padding-top:0.4rem;border-top:1px solid #f1f5f9;">
                  <div style="font-size:0.73rem;font-weight:700;color:#475569;margin-bottom:0.3rem;">
                    Projected Shares · 2030
                  </div>
                """, unsafe_allow_html=True)
                for label, clr in [("High Skill", C_HIGH), ("Middle Skill", C_MID), ("Low Skill", C_LOW)]:
                    val = proj_shares[label][-1]
                    st.markdown(f"""
                    <div class="proj-row">
                      <span style="color:#64748b;">{label}</span>
                      <span style="color:{clr};font-weight:700;">{val:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            with st.container(key="card_yoy"):
                st.markdown("""
                <div class="chart-card-title">Year-on-Year Share Change (pp)</div>
                <div class="chart-card-sub">Annual change per skill tier</div>
                """, unsafe_allow_html=True)
                fig_yoy = go.Figure()
                for label, col, color in [
                    ("High Skill",   "High Share",   C_HIGH),
                    ("Middle Skill", "Middle Share",  C_MID),
                    ("Low Skill",    "Low Share",     C_LOW),
                ]:
                    yoy = adf[col].diff().values
                    fig_yoy.add_trace(go.Bar(
                        x=adf["Year"], y=yoy, name=label,
                        marker=dict(color=color, opacity=0.85),
                    ))
                lay_yoy = plotly_theme(h=255, ml=45, mr=10, mt=4, mb=12)
                lay_yoy["yaxis"]["title"] = "Change (pp)"
                lay_yoy["barmode"] = "group"
                lay_yoy["yaxis"]["zeroline"] = True
                lay_yoy["yaxis"]["zerolinecolor"] = "#e2e8f0"
                lay_yoy["showlegend"] = True
                fig_yoy.update_layout(**lay_yoy)
                st.plotly_chart(fig_yoy, width='stretch', config=dict(displayModeBar=False))

    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# VIEW: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "clust":
    cluster_features = []
    for occ in occ_groups:
        vals   = adf[occ].values
        cagr   = ((vals[-1] / vals[0]) ** (1 / (len(vals) - 1)) - 1) * 100 if vals[0] > 0 else 0
        mean_v = vals.mean()
        cov    = vals.std() / mean_v if mean_v > 0 else 0
        v2020  = adf.loc[adf["Year"] == 2020, occ].values[0]
        v2022  = adf.loc[adf["Year"] == 2022, occ].values[0]
        rec    = (v2022 - v2020) / v2020 * 100 if v2020 > 0 else 0
        cluster_features.append({
            "Occupation": occ,
            "CAGR %": cagr,
            "Volatility": cov,
            "Post-COVID Rec %": rec,
            "Avg Employed (K)": mean_v,
        })

    df_cl = pd.DataFrame(cluster_features)
    feats  = ["CAGR %", "Volatility", "Post-COVID Rec %"]
    sc     = StandardScaler()
    X      = sc.fit_transform(df_cl[feats])
    km     = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cl["Cluster"] = km.fit_predict(X)
    cm     = df_cl.groupby("Cluster")["CAGR %"].mean().sort_values()
    lmap   = {
        cm.index[0]: "Declining",
        cm.index[1]: "Stable",
        cm.index[2]: "High Growth",
    }
    df_cl["Category"] = df_cl["Cluster"].map(lmap)

    cat_color = {
        "High Growth": "#10b981",
        "Stable":      "#64748b",
        "Declining":   "#ef4444",
    }

    col_l, col_r = st.columns([3, 2], gap="small")

    with col_l:
        p1, p2 = st.columns(2, gap="small")
        with p1:
            if st.button("K-Means Clusters", key="btn_kmeans", use_container_width=True):
                st.session_state.clust_tab = "scatter"
                st.rerun()
        with p2:
            if st.button("CAGR % by Occupation", key="btn_cagr", use_container_width=True):
                st.session_state.clust_tab = "bar"
                st.rerun()

        active_clust_tab = st.session_state.clust_tab
        st.markdown(f"""
        <style>
        /* Shift only the inner buttons row up to align with right card top */
        [data-testid="stColumn"] [data-testid="stHorizontalBlock"]:has(.st-key-btn_kmeans) {{
            margin-top: -0.8rem !important;
            margin-bottom: 0 !important;
        }}
        /* Button styling */
        .st-key-btn_kmeans button,
        .st-key-btn_cagr button {{
            background: #ffffff !important;
            border: 1px solid #dde3ef !important;
            color: #64748b !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.72rem !important;
            padding: 0.25rem 0.5rem !important;
            transition: all 0.15s !important;
            box-shadow: none !important;
            height: 28px !important;
            min-height: 28px !important;
            line-height: 1.2 !important;
        }}
        .st-key-btn_{'kmeans' if active_clust_tab == 'scatter' else 'cagr'} button {{
            background: #162040 !important;
            color: #ffffff !important;
            border-color: #162040 !important;
        }}
        /* Align card bottoms by making columns stretch using flexbox */
        div[data-testid="stColumn"]:has(.st-key-card_kmeans) > div,
        div[data-testid="stColumn"]:has(.st-key-card_cagr) > div,
        div[data-testid="stColumn"]:has(.st-key-card_detail) > div {{
            height: 100% !important;
        }}

        div[data-testid="stColumn"]:has(.st-key-card_kmeans) [data-testid="stVerticalBlock"],
        div[data-testid="stColumn"]:has(.st-key-card_cagr) [data-testid="stVerticalBlock"],
        div[data-testid="stColumn"]:has(.st-key-card_detail) [data-testid="stVerticalBlock"] {{
            height: 100% !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        /* Left column - second child is the card itself */
        div[data-testid="stColumn"]:has(.st-key-card_kmeans) [data-testid="stVerticalBlock"] > div:has(.st-key-card_kmeans) {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}
        div[data-testid="stColumn"]:has(.st-key-card_cagr) [data-testid="stVerticalBlock"] > div:has(.st-key-card_cagr) {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        /* Right column - card_detail stretches to fill the height */
        div[data-testid="stColumn"]:has(.st-key-card_detail) [data-testid="stVerticalBlock"] > div:has(.st-key-card_detail) {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}

        .st-key-card_kmeans,
        .st-key-card_cagr,
        .st-key-card_detail {{
            flex-grow: 1 !important;
            height: 100% !important;
        }}
        .st-key-card_kmeans,
        .st-key-card_cagr {{
            margin-top: -0.65rem !important;
            padding-bottom: 0.8rem !important;
            margin-bottom: 0.8rem !important;
        }}
        
        .st-key-card_detail {{
            margin-top: -0.8rem !important;
            padding: 0.9rem 1.2rem 0.9rem 1.2rem !important;
            margin-bottom: 0.8rem !important;
        }}
        .st-key-card_cagr {{
            height: 342px !important;
            min-height: 342px !important;
            max-height: 342px !important;
            overflow-y: auto !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        cmap = {
            "High Growth": "#10b981",
            "Stable":      "#64748b",
            "Declining":   "#ef4444",
        }

        if active_clust_tab == "scatter":
            with st.container(key="card_kmeans"):
                st.markdown("""
                <div class="chart-card-title">K-Means Occupation Clusters</div>
                <div class="chart-card-sub">CAGR %  vs.  Volatility  ·  Bubble size = Avg Employment</div>
                """, unsafe_allow_html=True)

                fig_cl = px.scatter(
                    df_cl, x="CAGR %", y="Volatility",
                    color="Category", size="Avg Employed (K)",
                    hover_name="Occupation",
                    color_discrete_map=cmap,
                )
                fig_cl.update_traces(
                    marker=dict(line=dict(width=2, color="#ffffff"), sizemin=10, opacity=0.9),
                )
                lay_cl = plotly_theme(h=260, mb=32)
                fig_cl.update_layout(**lay_cl)
                st.plotly_chart(fig_cl, width='stretch', config=dict(displayModeBar=False))

        else:
            with st.container(key="card_cagr"):
                st.markdown("""
                <div class="chart-card-title">CAGR % by Occupation (2016 – 2025)</div>
                <div class="chart-card-sub">Compound Annual Growth Rate — all major groups</div>
                """, unsafe_allow_html=True)

                df_sorted  = df_cl.sort_values("CAGR %")
                bar_cols   = [cmap.get(c, NAVY) for c in df_sorted["Category"]]
                short_occs = [o.split(" and ")[0][:24] for o in df_sorted["Occupation"]]
                n_bars     = len(short_occs)
                bar_h      = max(240, n_bars * 24)
                fig_bar = go.Figure(go.Bar(
                    x=df_sorted["CAGR %"],
                    y=short_occs,
                    orientation="h",
                    marker=dict(color=bar_cols, opacity=0.88, line=dict(width=0)),
                ))
                lay_bar = plotly_theme(h=bar_h, ml=10, mr=20, mt=4, mb=12)
                lay_bar["xaxis"]["zeroline"] = True
                lay_bar["xaxis"]["zerolinecolor"] = "#e2e8f0"
                lay_bar["yaxis"]["tickfont"] = dict(size=10, color="#475569")
                lay_bar["showlegend"] = False
                fig_bar.update_layout(**lay_bar)
                st.plotly_chart(fig_bar, width='stretch', config=dict(displayModeBar=False))

    with col_r:
        with st.container(key="card_detail"):
            hdr_l, hdr_r = st.columns([3, 2])
            with hdr_r:
                cf = st.selectbox(
                    "Inspect Cluster",
                    ["High Growth", "Stable", "Declining"],
                    index=["High Growth", "Stable", "Declining"].index(st.session_state.cluster_filter),
                    key="cluster_inspect_dropdown",
                    on_change=update_cluster_filter,
                    label_visibility="collapsed",
                )
                st.session_state.cluster_filter = cf

            clr   = cat_color.get(cf, NAVY)
            subset = df_cl[df_cl["Category"] == cf].copy()

            with hdr_l:
                st.markdown(f"""
                <div class="chart-card-title">Cluster Detail</div>
                """, unsafe_allow_html=True)

            detail_html = f"""
            <div style="max-height: 272px; overflow-y: auto; padding-right: 0.2rem; margin-top: 0.2rem;">
            """
            for _, row in subset.iterrows():
                short    = row["Occupation"][:36]
                cagr_v   = row["CAGR %"]
                vol      = row["Volatility"]
                postcov  = row["Post-COVID Rec %"]
                bar_w    = max(4, min(100, int(((cagr_v + 5) / 15) * 100)))
                cagr_fmt = f"{cagr_v:+.2f}%"
                vol_fmt  = f"{vol:.3f}"
                cov_fmt  = f"{postcov:+.1f}%"
                detail_html += (
                    '<div class="occ-card" style="margin-bottom:0.35rem;padding:0.5rem 0.75rem;">'
                    '<div class="occ-row2">'
                    f'<span class="occ-name" style="font-size:0.73rem;">{short}</span>'
                    f'<span class="occ-cagr" style="color:{clr};font-size:0.73rem;">{cagr_fmt}</span>'
                    '</div>'
                    '<div class="occ-bar-bg" style="margin-top:0.25rem;">'
                    f'<div style="width:{bar_w}%;height:100%;border-radius:999px;background:{clr};"></div>'
                    '</div>'
                    '<div class="occ-meta" style="font-size:0.63rem;margin-top:0.15rem;">'
                    f'Volatility: <strong style="color:#475569;">{vol_fmt}</strong>'
                    f'&nbsp;·&nbsp; Post-COVID: <strong style="color:#475569;">{cov_fmt}</strong>'
                    '</div>'
                    '</div>'
                )
            detail_html += "</div>"
            st.markdown(detail_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
