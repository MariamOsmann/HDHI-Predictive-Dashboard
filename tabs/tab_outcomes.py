# tabs/tab_outcomes.py
import streamlit as st
import plotly.express as px
from config import DURATION_ICU_COL

def render_tab(train, disease_cols):
    st.header("Clinical Outcomes 🏥")

    if DURATION_ICU_COL not in train.columns:
        st.warning("ICU duration column not found")
        return

    data = []
    for d in disease_cols:
        avg = train[train[d]==1][DURATION_ICU_COL].mean()
        data.append({"Disease": d, "Avg_Duration": avg})

    df = px.bar(
        data,
        x="Disease",
        y="Avg_Duration",
        title="Average ICU Duration per Disease"
    )

    st.plotly_chart(df, use_container_width=True)