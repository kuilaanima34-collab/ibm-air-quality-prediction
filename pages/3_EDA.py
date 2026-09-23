"""
pages/3_EDA.py
--------------
Exploratory Data Analysis:
  - Time-series trends for selected pollutants
  - Pollutant distribution histograms
  - Correlation heatmap
  - AQI level pie chart
  - Hour-of-day and month-of-year average patterns
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import plotly.express as px
import streamlit as st

from sidebar_style import (
    apply_sidebar_style, DATA_PATH, AQI_COLOURS, FEATURE_COLUMNS,
)

st.set_page_config(page_title="EDA | Air Quality", page_icon="🔬", layout="wide")
apply_sidebar_style()

st.title("🔬 Exploratory Data Analysis")
st.markdown("Visualise patterns, distributions and correlations across the dataset.")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
@st.cache_data
def get_df():
    import train_model as tm
    return tm.load_and_clean(DATA_PATH)

df = get_df()
LABEL_NAME_MAP = {0: "Good", 1: "Moderate", 2: "Unhealthy", 3: "Hazardous"}
df["AQI_Label"] = df["AQI_Level"].map(LABEL_NAME_MAP)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Time Series", "Distributions", "Correlation", "AQI Breakdown", "Temporal Patterns"
])

# ── Tab 1: Time Series ─────────────────────────────────────────────────────
with tab1:
    st.subheader("Pollutant Trends Over Time")
    cols_to_plot = st.multiselect(
        "Select sensors to plot",
        options=["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH", "AH"],
        default=["CO(GT)", "C6H6(GT)", "NOx(GT)"],
    )
    if cols_to_plot:
        # Downsample every 4 rows to keep chart fast
        plot_df = df[["DateTime"] + cols_to_plot].iloc[::4].reset_index(drop=True)
        fig = px.line(
            plot_df, x="DateTime", y=cols_to_plot,
            title="Sensor Readings Over Time",
            template="plotly_white",
            labels={"value": "Value", "variable": "Sensor"},
        )
        fig.update_layout(height=420, legend_title_text="Sensor")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one sensor above.")

# ── Tab 2: Distributions ───────────────────────────────────────────────────
with tab2:
    st.subheader("Pollutant Distribution")
    col_sel = st.selectbox(
        "Select column",
        ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"],
    )
    d1, d2 = st.columns(2)
    with d1:
        fig_hist = px.histogram(
            df, x=col_sel, nbins=60, marginal="box",
            title=f"Distribution of {col_sel}",
            template="plotly_white",
            color_discrete_sequence=["#3b82d4"],
        )
        fig_hist.update_layout(height=380)
        st.plotly_chart(fig_hist, use_container_width=True)
    with d2:
        fig_box = px.box(
            df, y=col_sel, x="AQI_Label",
            color="AQI_Label", color_discrete_map=AQI_COLOURS,
            title=f"{col_sel} by AQI Level",
            template="plotly_white",
        )
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ── Tab 3: Correlation ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Feature Correlation Heatmap")
    corr_cols = [
        "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)",
        "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH",
    ]
    corr = df[corr_cols].corr().round(2)
    fig_hm = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Matrix",
        template="plotly_white",
        zmin=-1, zmax=1,
    )
    fig_hm.update_layout(height=480)
    st.plotly_chart(fig_hm, use_container_width=True)

    # Top correlations with C6H6
    st.subheader("Top Correlations with C6H6(GT)")
    c6_corr = corr["C6H6(GT)"].drop("C6H6(GT)").sort_values(ascending=False)
    fig_bar = px.bar(
        x=c6_corr.values, y=c6_corr.index, orientation="h",
        title="Pearson r with Benzene (C6H6)",
        template="plotly_white",
        color=c6_corr.values,
        color_continuous_scale="RdBu_r",
        labels={"x": "Pearson r", "y": "Feature"},
    )
    fig_bar.update_layout(height=340, coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Tab 4: AQI Breakdown ───────────────────────────────────────────────────
with tab4:
    st.subheader("AQI Level Distribution")
    a1, a2 = st.columns(2)
    with a1:
        counts = df["AQI_Label"].value_counts()
        fig_pie = px.pie(
            values=counts.values, names=counts.index,
            color=counts.index, color_discrete_map=AQI_COLOURS,
            title="AQI Level Share",
            template="plotly_white",
        )
        fig_pie.update_layout(height=380)
        st.plotly_chart(fig_pie, use_container_width=True)
    with a2:
        counts_df = counts.reset_index()
        counts_df.columns = ["AQI Level", "Count"]
        fig_count = px.bar(
            counts_df, x="AQI Level", y="Count",
            color="AQI Level", color_discrete_map=AQI_COLOURS,
            title="Record Count per AQI Level",
            template="plotly_white", text_auto=True,
        )
        fig_count.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_count, use_container_width=True)

# ── Tab 5: Temporal Patterns ───────────────────────────────────────────────
with tab5:
    st.subheader("Hourly & Monthly Averages")
    col_t = st.selectbox("Select sensor", ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"],
                          key="temp_col")
    t1, t2 = st.columns(2)
    with t1:
        hourly = df.groupby("Hour")[col_t].mean().reset_index()
        fig_h = px.line(
            hourly, x="Hour", y=col_t,
            title=f"Average {col_t} by Hour of Day",
            template="plotly_white",
            markers=True,
        )
        fig_h.update_layout(height=340)
        st.plotly_chart(fig_h, use_container_width=True)
    with t2:
        monthly = df.groupby("Month")[col_t].mean().reset_index()
        fig_m = px.bar(
            monthly, x="Month", y=col_t,
            title=f"Average {col_t} by Month",
            template="plotly_white",
            color=col_t,
            color_continuous_scale="Oranges",
        )
        fig_m.update_layout(height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_m, use_container_width=True)
