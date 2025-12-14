# utils/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st

def clean_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace('-', '_', regex=False)
        .str.replace('\.', '_', regex=True)
        .str.replace(' ', '_', regex=False)
        .str.upper()
    )
    return df


def auto_detect_binary_columns(df):
    bin_cols = []
    for col in df.columns:
        vals = df[col].dropna().unique()
        normalized = set(str(v).strip().lower() for v in vals)
        if normalized.issubset({'0', '1', 'true', 'false', 'yes', 'no', 't', 'f'}) and len(normalized) <= 2:
            bin_cols.append(col)
    return bin_cols


def normalize_binary_columns(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(
                lambda x: 1 if str(x).lower() in ('1','true','yes','t')
                else 0 if str(x).lower() in ('0','false','no','f')
                else np.nan
            )
    return df


@st.cache_resource
def encode_columns(train, val, test, cols):
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        all_vals = pd.concat(
            [df[col] for df in [train, val, test] if df is not None and col in df],
            axis=0
        ).astype(str).fillna("NA")

        le.fit(all_vals.unique())

        for df in [train, val, test]:
            if df is not None and col in df:
                df[col] = le.transform(df[col].astype(str).fillna("NA"))

        encoders[col] = le

    return encoders, train, val, test


@st.cache_resource
def fill_numeric_with_median(train, val, test):
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    medians = train[numeric_cols].median()

    train[numeric_cols] = train[numeric_cols].fillna(medians)
    if val is not None and not val.empty:
        val[numeric_cols] = val[numeric_cols].fillna(medians)
    if test is not None and not test.empty:
        test[numeric_cols] = test[numeric_cols].fillna(medians)

    return medians, train, val, test
