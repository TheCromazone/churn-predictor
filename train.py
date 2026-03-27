"""
train.py — End-to-end training pipeline for the customer churn predictor.

Usage:
    python train.py                       # train with defaults
    python train.py --tune                # run Optuna hyperparameter search
    python train.py --data path/to/data.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import build_feature_pipeline
from src.evaluate import plot_roc_curve, plot_confusion_matrix, plot_shap_importance

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/telcochurn.csv")
MODEL_OUT = Path("models/xgb_churn.pkl")
RANDOM_STATE = 42


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    # Target: 1 = churned, 0 = retained
    y = (df["Churn"].str.lower() == "yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    logger.info(f"Dataset: {len(X):,} rows | Churn rate: {y.mean():.1%}")
    return X, y


# ─── Default XGBoost Config ───────────────────────────────────────────────────
DEFAULT_XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3,  # accounts for class imbalance
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ─── Hyperparameter Tuning (Optuna) ─────────────────────────────────────────────────────────────────
def tune_hyperparams(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    feature_pipeline = build_feature_pipeline()
    X_transformed = feature_pipeline.fit_transform(X)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(model, X_transformed, y, cv=cv, scoring="roc_auc")
        return scores["test_score"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best AUC: {study.best_value:.4f}")
    return study.best_params


# ─── Training ─────────────────────────────────────────────────────────────────
def train(data_path: Path, tune: bool = False) -> Pipeline:
    X, y = load_data(data_path)
    feature_pipeline = build_feature_pipeline()

    xgb_params = tune_hyperparams(X, y) if tune else DEFAULT_XGB_PARAMS
    classifier = XGBClassifier(**xgb_params)

    full_pipeline = Pipeline([
        ("features", feature_pipeline),
        ("classifier", classifier),
    ])

    # ── Cross-validation ──────────────────────────────────────────────────────
    logger.info("Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAndom_STATE)
    cv_results = cross_validate(
        full_pipeline, X, y, cv=cv,
        scoring=["accuracy", "roc_auc", "f1"],
        return_train_score=True,
    )

    logger.info("── Cross-Validation Results ──────────────────────────")
    for metric in ["accuracy", "roc_auc", "f1"]:
        test_scores = cv_results[f"test_{metric}"]
        logger.info(f"  {metric:12s}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")

    # ── Final fit on full training data ───────────────────────────────────────
    logger.info("Fitting final model on full dataset...")
    full_pipeline.fit(X, y)

    # ── Save model ────────────────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(full_pipeline, f)
    logger.info(f"Model saved to {MODEL_OUT}")

    # ── Evaluation plots ──────────────────────────────────────────────────────
    y_pred_proba = full_pipeline.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    logger.info("\n" + classification_report(y, y_pred, target_names=["Retained", "Churned"]))
    plot_roc_curve(y, y_pred_proba)
    plot_confusion_matrix(y, y_pred)

    return full_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn prediction model")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    args = parser.parse_args()
    train(args.data, args.tune)
