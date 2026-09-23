"""
pages/4_Batch_Predict.py
------------------------
Batch prediction page:
  - User uploads a CSV of sensor readings
  - App predicts C6H6 and AQI level for every row
  - Results downloadable as CSV
  - Summary charts shown inline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import io
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import streamlit as st

from sidebar_style import (
    apply_sidebar_style, models_exist, MODELS_DIR,
    AQI_COLOURS, FEATURE_COLUMNS, LABEL_MAP,
)

st.set_page_config(page_title="Batch Predict | Air Quality", page_icon="📁", layout="wide")
apply_sidebar_style()

st.title("📁 Batch Prediction")
st.markdown(
    "Upload a CSV containing sensor readings. "
    "The app will predict **C6H6 concentration** and **AQI level** for each row."
)

if not models_exist():
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    bundle = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return bundle["regressor"], bundle["classifier"], scaler

reg, clf, scaler = load_models()

# ---------------------------------------------------------------------------
# Template download
# ---------------------------------------------------------------------------
st.subheader("1. Download the input template")
template_df = pd.DataFrame(columns=FEATURE_COLUMNS)
# Pre-fill one example row
template_df.loc[0] = [2.0, 1300.0, 150.0, 100.0, 1400.0, 1000.0, 15.0, 50.0, 0.75, 12, 0, 6]
csv_template = template_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download CSV Template",
    data=csv_template,
    file_name="air_quality_template.csv",
    mime="text/csv",
)

st.divider()

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
st.subheader("2. Upload your CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    try:
        input_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.markdown(f"**Uploaded:** {uploaded.name} — {len(input_df)} rows")
    with st.expander("Preview uploaded data"):
        st.dataframe(input_df.head(20), use_container_width=True)

    # Check required columns
    missing_cols = [c for c in FEATURE_COLUMNS if c not in input_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.info("Download the template above to see the expected format.")
        st.stop()

    # ---------------------------------------------------------------------------
    # Run predictions
    # ---------------------------------------------------------------------------
    with st.spinner("Running predictions …"):
        X = input_df[FEATURE_COLUMNS].values.astype(float)
        X_sc = scaler.transform(X)

        c6h6_preds   = reg.predict(X_sc)
        level_codes  = clf.predict(X_sc)
        level_probas = clf.predict_proba(X_sc)

        result_df = input_df.copy()
        result_df["C6H6_Predicted"]  = c6h6_preds.round(3)
        result_df["AQI_Level_Code"]  = level_codes
        result_df["AQI_Level_Label"] = [LABEL_MAP[int(c)] for c in level_codes]
        for i, label in LABEL_MAP.items():
            result_df[f"Prob_{label}"] = level_probas[:, i].round(4)

    st.success(f"Predictions complete for {len(result_df)} rows.")

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------
    st.subheader("3. Results")
    show_cols = FEATURE_COLUMNS + ["C6H6_Predicted", "AQI_Level_Label"]
    st.dataframe(result_df[show_cols], use_container_width=True)

    # Download button
    out_csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Results CSV",
        data=out_csv,
        file_name="air_quality_predictions.csv",
        mime="text/csv",
    )

    st.divider()

    # ---------------------------------------------------------------------------
    # Summary charts
    # ---------------------------------------------------------------------------
    st.subheader("4. Prediction Summary")
    v1, v2 = st.columns(2)

    with v1:
        fig_hist = px.histogram(
            result_df, x="C6H6_Predicted", nbins=40,
            title="Distribution of Predicted C6H6 Values",
            template="plotly_white",
            color_discrete_sequence=["#3b82d4"],
            labels={"C6H6_Predicted": "Predicted C6H6 (µg/m³)"},
        )
        fig_hist.update_layout(height=340)
        st.plotly_chart(fig_hist, use_container_width=True)

    with v2:
        aqi_counts = result_df["AQI_Level_Label"].value_counts().reset_index()
        aqi_counts.columns = ["AQI Level", "Count"]
        fig_bar = px.bar(
            aqi_counts, x="AQI Level", y="Count",
            color="AQI Level", color_discrete_map=AQI_COLOURS,
            title="AQI Level Distribution in Predictions",
            template="plotly_white", text_auto=True,
        )
        fig_bar.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # C6H6 over row index (useful if data is time-ordered)
    fig_line = px.line(
        result_df.reset_index(), x="index", y="C6H6_Predicted",
        color="AQI_Level_Label", color_discrete_map=AQI_COLOURS,
        title="Predicted C6H6 Across Rows",
        template="plotly_white",
        labels={"index": "Row", "C6H6_Predicted": "C6H6 (µg/m³)", "AQI_Level_Label": "AQI Level"},
    )
    fig_line.update_layout(height=360)
    st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("Upload a CSV file above to get started. Use the template if you need a starting point.")

    # Show expected columns as reference
    st.subheader("Expected Input Columns")
    col_info = pd.DataFrame({
        "Column":      FEATURE_COLUMNS,
        "Type":        ["float", "float", "float", "float", "float", "float",
                        "float", "float", "float", "int", "int", "int"],
        "Description": [
            "CO concentration mg/m³", "Tin oxide CO sensor",
            "NOx ppb", "NO2 µg/m³", "Tungsten oxide NO2 sensor",
            "Indium oxide O3 sensor", "Temperature °C",
            "Relative Humidity %", "Absolute Humidity g/m³",
            "Hour of day 0-23", "Day of week 0=Mon", "Month 1-12",
        ],
    })
    st.dataframe(col_info, use_container_width=True)
