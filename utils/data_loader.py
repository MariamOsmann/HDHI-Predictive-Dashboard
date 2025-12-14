# utils/data_loader.py
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_csv_if_exists(filename):
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Error loading CSV: {e}")
    return None
