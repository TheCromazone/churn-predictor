"""
predict.py — Inference interface for the trained churn model.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

RISK_TIERS = [
    (0.8, "HIGH"),
    (0.5, "MEDIUM"),
    (0.0, "LOW"),
]


class ChurnPredictor:
    """
    Thin wrapper around a trained sklearn Pipeline for churn inference.

    Usage:
        predictor = ChurnPredictor.load("models/xgb_churn.pkl")
        result = predictor.predict(customer_dict)
    """

    def __init__(self, pipeline) -> None:
        self._pipeline = pipeline

    @classmethod
    def load(cls, model_path: str | Path) -> "ChurnPredictor":
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        return cls(pipeline)

    def predict(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn risk for a single customer.

        Args:
            customer: dict of raw feature values matching the training schema.

        Returns:
            dict with:
              - churn_probability (float): probability of churn in [0, 1]
              - risk_tier (str): "HIGH", "MEDIUM", or "LOW"
              - will_churn (bool): True if probability >= 0.5
        """
        df = pd.DataFrame([customer])
        proba = self._pipeline.predict_proba(df)[0, 1]
        tier = next(t for threshold, t in RISK_TIERS if proba >= threshold)

        return {
            "churn_probability": round(float(proba), 4),
            "risk_tier": tier,
            "will_churn": proba >= 0.5,
        }

    def predict_batch(self, customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch predict for a list of customer dicts."""
        return [self.predict(c) for c in customers]

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add churn_probability and risk_tier columns to a DataFrame."""
        probas = self._pipeline.predict_proba(df)[:, 1]
        results = df.copy()
        results["churn_probability"] = probas
        results["risk_tier"] = [
            next(t for threshold, t in RISK_TIERS if p >= threshold)
            for p in probas
        ]
        results["will_churn"] = probas >= 0.5
        return results.sort_values("churn_probability", ascending=False)
