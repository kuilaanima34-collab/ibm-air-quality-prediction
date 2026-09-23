"""
pages/1_Home.py
---------------
Home page: live single-record prediction (regression + classification)
with interactive sliders and dataset summary statistics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from sidebar_style import (
    apply_sidebar_style, load_metadata, models_exist,
    MODELS_DIR, DATA_PATH, AQI_COLOURS, FEATURE_COLUMNS, LABEL_MAP,
)

st.set_page_config(page_title="Home | Air Quality", page_icon="🏠", layout="wide")
apply_sidebar_style()

st.title("🏠 Home — Live Prediction")
st.markdown("Adjust the sensor readings and click **Predict** to get instant results.")

# ---------------------------------------------------------------------------
# Guard: models must exist
# ---------------------------------------------------------------------------
if not models_exist():
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Load models (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    bundle = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return bundle["regressor"], bundle["classifier"], scaler

reg, clf, scaler = load_models()
meta = load_metadata()

# ---------------------------------------------------------------------------
# Load dataset stats for slider ranges (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def get_df():
    import train_model as tm
    return tm.load_and_clean(DATA_PATH)

df = get_df()
stats = df[FEATURE_COLUMNS].describe()

def stat(col, s):
    return float(stats.loc[s, col])

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("predict_form"):
    st.markdown('<p class="section-hdr">Pollutant Sensors</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        co_gt      = st.slider("CO (mg/m³)",       0.0,  15.0, float(stats.loc["mean","CO(GT)"]),    step=0.1)
        pt08_s1    = st.slider("PT08.S1 CO",      600.0, 2500.0, float(stats.loc["mean","PT08.S1(CO)"]), step=10.0)
    with c2:
        nox_gt     = st.slider("NOx (ppb)",        0.0, 1500.0, float(stats.loc["mean","NOx(GT)"]),  step=5.0)
        no2_gt     = st.slider("NO2 (µg/m³)",      0.0,  350.0, float(stats.loc["mean","NO2(GT)"]),  step=2.0)
    with c3:
        pt08_s4    = st.slider("PT08.S4 NO2",     700.0, 2800.0, float(stats.loc["mean","PT08.S4(NO2)"]), step=10.0)
        pt08_s5    = st.slider("PT08.S5 O3",      200.0, 2700.0, float(stats.loc["mean","PT08.S5(O3)"]),  step=10.0)

    st.markdown('<p class="section-hdr">Environment</p>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        temp = st.slider("Temperature (°C)", -5.0, 45.0, float(stats.loc["mean","T"]),  step=0.5)
        rh   = st.slider("Relative Humidity (%)", 5.0, 100.0, float(stats.loc["mean","RH"]), step=1.0)
    with e2:
        ah   = st.slider("Absolute Humidity",  0.1, 2.5, float(stats.loc["mean","AH"]),  step=0.01)
    with e3:
        hour = st.slider("Hour of Day",    0, 23, 12)
        dow  = st.slider("Day of Week",    0,  6,  0)
        mon  = st.slider("Month",          1, 12,  6)

    predict_btn = st.form_submit_button("🔮 Predict Now", use_container_width=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if predict_btn:
    x = np.array([[co_gt, pt08_s1, nox_gt, no2_gt, pt08_s4, pt08_s5,
                   temp, rh, ah, hour, dow, mon]])
    x_sc = scaler.transform(x)

    c6h6  = float(reg.predict(x_sc)[0])
    code  = int(clf.predict(x_sc)[0])
    proba = clf.predict_proba(x_sc)[0]
    level = LABEL_MAP[code]
    colour = AQI_COLOURS[level]

    st.divider()
    st.subheader("Prediction Results")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Predicted C6H6", f"{c6h6:.2f} µg/m³")
        lvl_c6h6 = "Low" if c6h6 < 5 else "Moderate" if c6h6 < 10 else "High" if c6h6 < 20 else "Very High"
        col_c6h6 = "#22c55e" if c6h6 < 5 else "#eab308" if c6h6 < 10 else "#f97316" if c6h6 < 20 else "#ef4444"
        st.markdown(f'<span class="aqi-badge" style="background:{col_c6h6}">{lvl_c6h6}</span>', unsafe_allow_html=True)

    with r2:
        st.metric("AQI Level", level)
        st.markdown(f'<span class="aqi-badge" style="background:{colour}">{level}</span>', unsafe_allow_html=True)

    with r3:
        proba_df = pd.DataFrame({"Level": list(LABEL_MAP.values()), "Probability": proba})
        fig = px.bar(proba_df, x="Level", y="Probability",
                     color="Level", color_discrete_map=AQI_COLOURS,
                     text_auto=".1%", template="plotly_white")
        fig.update_layout(height=260, showlegend=False,
                          yaxis_range=[0, 1], yaxis_tickformat=".0%",
                          margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Benzene gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=c6h6,
        number={"suffix": " µg/m³"},
        title={"text": "C6H6 Concentration"},
        gauge={
            "axis": {"range": [0, 40]},
            "bar":  {"color": col_c6h6},
            "steps": [
                {"range": [0,  5],  "color": "#dcfce7"},
                {"range": [5,  10], "color": "#fef9c3"},
                {"range": [10, 20], "color": "#ffedd5"},
                {"range": [20, 40], "color": "#fee2e2"},
            ],
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=40, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------------------------------------------------------
# Dataset summary
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Dataset Summary Statistics")
st.dataframe(df[FEATURE_COLUMNS + ["C6H6(GT)"]].describe().round(3), use_container_width=True)
