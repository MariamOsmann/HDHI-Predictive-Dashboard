# app.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from utils.data_loader import load_csv_if_exists
from utils.preprocessing import clean_columns, auto_detect_binary_columns
from models.train_models import train_outcome_model

from tabs import tab_outcomes, tab_forecasting, tab_patient
from config import OUTCOME_COL


st.set_page_config(layout="wide", page_title="HDHI Predictive Dashboard 🏥")


def main():
    st.title("HDHI Predictive Dashboard 🏥")

    # ---------- Load data ----------
    train = load_csv_if_exists("data/HDHI_Train.csv")
    if train is None:
        st.error("❌ HDHI_Train.csv not found")
        return

    train = clean_columns(train)

    if OUTCOME_COL not in train.columns:
        st.error(f"❌ Outcome column '{OUTCOME_COL}' not found")
        return


    disease_cols = auto_detect_binary_columns(train)


    with st.sidebar:
        st.header("⚙ Forecast Settings")

        # Detect possible date columns
        date_like_cols = [
            c for c in train.columns
            if any(k in c.lower() for k in ["date", "time", "day"])
        ]

        date_col = st.selectbox(
            "Select Date Column",
            options=[""] + train.columns.tolist(),
            format_func=lambda x: f"🕒 {x}" if x in date_like_cols else x
        )



    le_outcome = LabelEncoder()
    train[OUTCOME_COL] = train[OUTCOME_COL].astype(str).fillna("NA")
    le_outcome.fit(train[OUTCOME_COL])

    y = le_outcome.transform(train[OUTCOME_COL])

    # ---------- Features ----------
    X = train.drop(columns=[OUTCOME_COL], errors="ignore")
    X = X.select_dtypes(include=np.number)

    if X.empty:
        st.error("❌ No numeric features available for training")
        return

    # ---------- Train model ----------
    with st.spinner("⏳ Training Outcome model..."):
        bundle = train_outcome_model(X, y)

        model = bundle["model"]
        feature_names = bundle["feature_names"]
        medians = bundle["medians"]
        feature_importances = bundle["feature_importances"]

    st.success("✅ Model trained successfully")

    # ---------- Tabs ----------
    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Clinical Outcomes",
        "2️⃣ Forecasting",
        "3️⃣ Patient Prediction"
    ])

    with tab1:
        tab_outcomes.render_tab(train, disease_cols)


    with tab2:
        tab_forecasting.render_tab(
            train=train,
            disease_cols=disease_cols,
            date_col=date_col
        )

    with tab3:
        tab_patient.render_tab(
            model=model,
            feature_names=feature_names,
            encoders={},
            medians=medians,
            le_outcome=le_outcome,
            X_test=None
        )


if __name__ == "_main_":
    main()