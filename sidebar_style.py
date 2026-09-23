"""
sidebar_style.py
----------------
Shared CSS injection and sidebar branding used by every page.
Import and call apply_sidebar_style() at the top of each page.
"""

import json
import os
import streamlit as st

# ---------------------------------------------------------------------------
# Constants (re-exported so pages don't need to duplicate them)
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
DATA_PATH  = os.path.join("data", "Air Quality.csv")

AQI_COLOURS = {
    "Good":      "#22c55e",
    "Moderate":  "#eab308",
    "Unhealthy": "#f97316",
    "Hazardous": "#ef4444",
}

FEATURE_COLUMNS = [
    "CO(GT)", "PT08.S1(CO)", "NOx(GT)", "NO2(GT)",
    "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    "Hour", "DayOfWeek", "Month",
]

LABEL_MAP = {0: "Good", 1: "Moderate", 2: "Unhealthy", 3: "Hazardous"}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
}

/* AQI badge */
.aqi-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 24px;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.03em;
    margin-top: 6px;
}

/* Metric card */
.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
}

/* Section header */
.section-hdr {
    font-size: 1rem;
    font-weight: 600;
    color: #334155;
    border-left: 4px solid #3b82f6;
    padding-left: 8px;
    margin: 16px 0 8px 0;
}
</style>
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def apply_sidebar_style():
    """Inject CSS and render sidebar branding. Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("## 🌫️ Air Quality")
        st.markdown("**Prediction System**")
        st.divider()
        st.caption("UCI Air Quality Dataset\nMar 2004 – Apr 2005\n9,357 hourly records")
        st.divider()


def load_metadata() -> dict:
    path = os.path.join(MODELS_DIR, "metadata.json")
    if not os.path.exists(path):
        st.error("metadata.json not found. Run `python train_model.py` first.")
        st.stop()
    with open(path) as f:
        return json.load(f)


def models_exist() -> bool:
    return (
        os.path.exists(os.path.join(MODELS_DIR, "model.pkl")) and
        os.path.exists(os.path.join(MODELS_DIR, "scaler.pkl"))
    )
