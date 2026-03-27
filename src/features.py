"""
features.py — Feature engineering pipeline for churn prediction.

Engineered features:
  - charge_per_month_tenure:  monthly charges normalized by tenure
  - high_value_at_risk:       flag for high-spend, short-tenure customers
  - services_count:           number of add-on services subscribed
  - support_ticket_rate:      tickets per month of tenure
  - is_new_customer:          tenure <= 3 months
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ─── Numeric columns ──────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "NumAdminTickets", "NumTechTickets",
]

# ─── Categorical columns ─────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# ─── Service columns (for aggregation) ───────────────────────────────────────
SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that adds domain-specific engineered features
    before the main preprocessing pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Convert TotalCharges to numeric (sometimes loaded as string)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # Ticket counts if not already present
        if "NumAdminTickets" not in df.columns:
            df["NumAdminTickets"] = 0
        if "NumTechTickets" not in df.columns:
            df["NumTechTickets"] = 0

        tenure = df["tenure"].clip(lower=1)  # avoid division by zero

        # Engineered features
        df["charge_per_month"] = df["MonthlyCharges"] / tenure
        df["support_ticket_rate"] = (
            df["NumAdminTickets"] + df["NumTechTickets"]
        ) / tenure
        df["services_count"] = df[SERVICE_COLS].apply(
            lambda row: (row == "Yes").sum(), axis=1
        )
        df["is_new_customer"] = (df["tenure"] <= 3).astype(int)
        df["high_value_at_risk"] = (
            (df["MonthlyCharges"] > 70) & (df["tenure"] <= 12)
        ).astype(int)
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

        return df


def build_feature_pipeline() -> Pipeline:
    """
    Build and return the full sklearn preprocessing + feature engineering pipeline.
    """
    engineer = ChurnFeatureEngineer()

    all_numeric = NUMERIC_COLS + [
        "charge_per_month", "support_ticket_rate",
        "services_count", "is_new_customer",
        "high_value_at_risk", "is_month_to_month",
    ]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, all_numeric),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",
    )

    return Pipeline([
        ("engineer", engineer),
        ("preprocessor", preprocessor),
    ])
