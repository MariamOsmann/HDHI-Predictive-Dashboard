# tabs/tab_forecasting.py
import streamlit as st
import pandas as pd

from utils.plotting import generate_time_series_plot
from config import EMERGENCY_COL

def render_tab(train, disease_cols, date_col):
    st.header("Operational Forecasting 📊")

    # ---------- Guard ----------
    if not date_col:
        st.warning("⚠️ Please select a date column from the sidebar")
        return

    if date_col not in train.columns:
        st.error("❌ Selected date column not found in data")
        return

    # ---------- Ensure datetime ----------
    df = train.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # ---------- Controls ----------
    freq = st.selectbox(
        "Frequency",
        ["D", "W", "M"],
        index=1
    )

    periods = st.number_input(
        "Forecast periods",
        min_value=1,
        max_value=52,
        value=4
    )

    run = st.button("Run Forecast")

    # ---------- Emergency Forecast ----------
    if EMERGENCY_COL in df.columns:
        fig = generate_time_series_plot(
            df,
            date_col,
            EMERGENCY_COL,
            freq=freq,
            title="Emergency Admissions",
            forecast_periods=periods,
            run_forecast=run
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Disease Forecast ----------
    disease = st.selectbox(
        "Select Disease",
        options=disease_cols
    )

    if disease in df.columns:
        fig = generate_time_series_plot(
            df,
            date_col,
            disease,
            freq=freq,
            title=f"{disease} Cases",
            forecast_periods=periods,
            run_forecast=run
        )
        st.plotly_chart(fig, use_container_width=True)