# tabs/tab_patient.py
import streamlit as st
import pandas as pd
import numpy as np

def render_tab(
    model,
    feature_names,
    encoders,
    medians,
    le_outcome,
    X_test
):
    st.header("Individual Patient Outcome Prediction 👤")

    if model is None:
        st.warning("⚠️ Model not trained")
        return

    # -------- Manual input --------
    st.subheader("Manual Patient Input")

    inputs = {}

    for f in feature_names[:15]:
        if f in encoders:
            le = encoders[f]
            inputs[f] = st.selectbox(f, le.classes_)
        else:
            inputs[f] = st.number_input(
                f,
                value=float(medians.get(f, 0.0))
            )

    if st.button("Predict Outcome"):
        df = pd.DataFrame([inputs])

        # Encode categoricals if exist
        for c in df.columns:
            if c in encoders:
                df[c] = encoders[c].transform(df[c])

        # Fill missing features
        for f in feature_names:
            if f not in df:
                df[f] = medians.get(f, 0)

        df = df[feature_names]

        # ---------- Prediction ----------
        proba = model.predict_proba(df)[0]
        pred = model.predict(df)[0]

        # ---------- SAFE inverse_transform ----------
        if le_outcome is not None:
            label = le_outcome.inverse_transform([pred])[0]
        else:
            label = str(pred)

        st.success(f"✅ Predicted Outcome: {label}")

        prob_df = pd.DataFrame({
            "Outcome": le_outcome.classes_ if le_outcome else list(range(len(proba))),
            "Probability (%)": np.round(proba * 100, 2)
        })

        st.dataframe(prob_df, use_container_width=True)