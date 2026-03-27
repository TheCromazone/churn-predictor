"""
evaluate.py — Model evaluation plots and metrics for churn prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save: bool = True) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Customer Churn Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUTPUT_DIR / 'roc_curve.png'}")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save: bool = True) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Retained", "Churned"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Customer Churn Model", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.show()


def plot_shap_importance(model_pipeline, X: pd.DataFrame, n_top: int = 15, save: bool = True) -> None:
    """
    Generate SHAP summary plot for the XGBoost classifier inside the pipeline.
    """
    try:
        import shap
    except ImportError:
        print("Install shap: pip install shap")
        return

    classifier = model_pipeline.named_steps["classifier"]
    feature_step = model_pipeline.named_steps["features"]
    X_transformed = feature_step.transform(X)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values, X_transformed,
        max_display=n_top,
        show=False,
        plot_type="bar",
    )
    ax.set_title("Feature Importance (SHAP Values)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUTPUT_DIR / 'shap_importance.png'}")
    plt.show()
