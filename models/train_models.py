# models/train_models.py
import xgboost as xgb
import pandas as pd
import numpy as np


def train_outcome_model(X, y):
    """
    Train outcome prediction model using XGBoost.

    Returns:
        dict: model bundle containing:
            - model
            - feature_names
            - medians
            - feature_importances
    """

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )

    model.fit(X, y)

    # Feature metadata
    feature_names = X.columns.tolist()
    medians = X.median()

    # Safe feature importance
    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.zeros(len(feature_names))

    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "feature_names": feature_names,
        "medians": medians,
        "feature_importances": fi_df
    }


def train_disease_models(train, disease_cols, outcome_col):
    """
    Train binary disease prediction models.
    """
    disease_models = {}

    for d in disease_cols:
        if d not in train.columns:
            continue

        X = train.drop(columns=[d, outcome_col], errors="ignore")
        y = train[d].fillna(0).astype(int)

        X = X.select_dtypes(include="number")
        if X.empty or y.nunique() < 2:
            continue

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False
        )
        model.fit(X, y)

        disease_models[d] = {
            "model": model,
            "features": X.columns.tolist()
        }

    return disease_models