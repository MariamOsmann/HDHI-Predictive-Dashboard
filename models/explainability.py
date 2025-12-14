# models/explainability.py
import shap
import matplotlib.pyplot as plt
import streamlit as st

def render_shap(model, X):
    sample = X.sample(min(500, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10,6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    st.pyplot(fig)