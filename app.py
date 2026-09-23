"""
app.py
------
Entry point for the Streamlit multi-page app.
Run:  streamlit run app.py

This page acts as the landing / welcome screen.
Navigation to the 4 feature pages is handled by Streamlit's
built-in pages/ directory support.
"""

import streamlit as st
from sidebar_style import apply_sidebar_style, load_metadata, models_exist

st.set_page_config(
    page_title="Air Quality Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.title("🌫️ Air Quality Prediction System")
st.markdown(
    "Predict **Benzene (C6H6) concentration** and **AQI pollution level** "
    "from metal-oxide chemical sensors using machine learning."
)
st.divider()

# ---------------------------------------------------------------------------
# Status cards
# ---------------------------------------------------------------------------
if models_exist():
    meta = load_metadata()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dataset Records", "9,357")
    with c2:
        st.metric("Regression R²", f"{meta.get('r2_score', 0):.4f}")
    with c3:
        st.metric("Classifier Accuracy", f"{meta.get('accuracy', 0)*100:.1f}%")
    with c4:
        st.metric("Features Used", str(len(meta.get("feature_columns", []))))
else:
    st.warning("Models not found. Run `python train_model.py` to train first.")

st.divider()

# ---------------------------------------------------------------------------
# Navigation cards
# ---------------------------------------------------------------------------
st.subheader("Navigate to a page")
n1, n2, n3, n4 = st.columns(4)

with n1:
    st.markdown("### 🏠 Home")
    st.markdown("Overview, live single-record prediction, dataset summary stats.")

with n2:
    st.markdown("### 📊 Model Insights")
    st.markdown("Feature importances, confusion matrix, regression performance charts.")

with n3:
    st.markdown("### 🔬 EDA")
    st.markdown("Time-series trends, correlation heatmap, pollutant distributions.")

with n4:
    st.markdown("### 📁 Batch Predict")
    st.markdown("Upload a CSV of sensor readings and download predictions.")

st.divider()

# ---------------------------------------------------------------------------
# Quick info
# ---------------------------------------------------------------------------
with st.expander("About the dataset"):
    st.markdown("""
The **UCI Air Quality Dataset** contains 9,357 hourly measurements recorded by a
multisensor device in an Italian city between **March 2004 and April 2005**.

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Records | 9,357 hourly rows |
| Missing sentinel | -200 (replaced with column medians) |
| Regression target | C6H6(GT) — Benzene concentration (µg/m³) |
| Classification target | AQI Level — Good / Moderate / Unhealthy / Hazardous |
""")
