"""
pages/2_Model_Insights.py
--------------------------
Model performance deep-dive:
  - Feature importance bar chart
  - Regression: actual vs predicted scatter + residual histogram
  - Classification: confusion matrix heatmap + per-class metrics table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, r2_score

from sidebar_style import (
    apply_sidebar_style, load_metadata, models_exist,
    MODELS_DIR, DATA_PATH, FEATURE_COLUMNS, LABEL_MAP,
)

st.set_page_config(page_title="Model Insights | Air Quality", page_icon="📊", layout="wide")
apply_sidebar_style()

st.title("📊 Model Insights")
st.markdown("Performance metrics, feature importances and error analysis for both models.")

if not models_exist():
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    bundle = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    return bundle["regressor"], bundle["classifier"], scaler

@st.cache_data
def get_df():
    import train_model as tm
    return tm.load_and_clean(DATA_PATH)

@st.cache_data
def get_split():
    from sklearn.model_selection import train_test_split
    df = get_df()
    X  = df[FEATURE_COLUMNS].values
    yr = df["C6H6(GT)"].values
    yc = df["AQI_Level"].values
    _, X_test, _, yr_test, _, yc_test = train_test_split(
        X, yr, yc, test_size=0.20, random_state=42
    )
    return X_test, yr_test, yc_test

reg, clf, scaler = load_models()
meta = load_metadata()
X_test, yr_test, yc_test = get_split()
X_test_sc = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Score cards
# ---------------------------------------------------------------------------
yr_pred = reg.predict(X_test_sc)
yc_pred = clf.predict(X_test_sc)
r2  = r2_score(yr_test, yr_pred)
acc = float((yc_pred == yc_test).mean())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Regression R²",     f"{r2:.4f}")
m2.metric("Classifier Accuracy", f"{acc*100:.1f}%")
m3.metric("Train samples", f"{meta.get('n_train',0):,}")
m4.metric("Test samples",  f"{meta.get('n_test',0):,}")
st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Feature Importance", "Regression Analysis", "Classification Analysis"])

# ── Tab 1: Feature Importance ──────────────────────────────────────────────
with tab1:
    fi = pd.read_csv(os.path.join(MODELS_DIR, "feature_importances.csv"))
    fig_fi = px.bar(
        fi, x="importance", y="feature", orientation="h",
        title="Feature Importance (Regressor — C6H6 Prediction)",
        template="plotly_white", color="importance",
        color_continuous_scale="Blues",
        labels={"importance": "Importance Score", "feature": "Feature"},
    )
    fig_fi.update_layout(height=420, coloraxis_showscale=False,
                         yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown(
        "Feature importance scores are from the **RandomForestRegressor** (mean decrease in impurity). "
        "Higher = more influential for predicting Benzene concentration."
    )

# ── Tab 2: Regression Analysis ─────────────────────────────────────────────
with tab2:
    reg_df = pd.DataFrame({"Actual": yr_test, "Predicted": yr_pred})
    reg_df["Residual"] = reg_df["Actual"] - reg_df["Predicted"]

    col_a, col_b = st.columns(2)
    with col_a:
        fig_scatter = px.scatter(
            reg_df, x="Actual", y="Predicted",
            title=f"Actual vs Predicted C6H6  (R²={r2:.4f})",
            template="plotly_white", opacity=0.5,
            color_discrete_sequence=["#3b82f6"],
            labels={"Actual": "Actual C6H6 (µg/m³)", "Predicted": "Predicted C6H6 (µg/m³)"},
        )
        # Perfect-fit reference line
        lo = float(min(yr_test.min(), yr_pred.min()))
        hi = float(max(yr_test.max(), yr_pred.max()))
        fig_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                              line=dict(color="red", dash="dash"))
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        fig_resid = px.histogram(
            reg_df, x="Residual", nbins=60,
            title="Residual Distribution",
            template="plotly_white",
            color_discrete_sequence=["#8b5cf6"],
            labels={"Residual": "Residual (Actual − Predicted)"},
        )
        fig_resid.update_layout(height=380)
        st.plotly_chart(fig_resid, use_container_width=True)

    st.dataframe(reg_df.describe().round(3), use_container_width=True)

# ── Tab 3: Classification Analysis ────────────────────────────────────────
with tab3:
    labels = list(LABEL_MAP.values())
    cm = confusion_matrix(yc_test, yc_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig_cm = px.imshow(
        cm_df, text_auto=True, color_continuous_scale="Blues",
        title="Confusion Matrix — AQI Level Classifier",
        labels={"x": "Predicted", "y": "Actual"},
        template="plotly_white",
    )
    fig_cm.update_layout(height=420)
    st.plotly_chart(fig_cm, use_container_width=True)

    # Per-class report
    report = classification_report(yc_test, yc_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.subheader("Classification Report")
    st.dataframe(report_df, use_container_width=True)
