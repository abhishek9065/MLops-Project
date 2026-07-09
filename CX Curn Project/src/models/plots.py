"""
Diagnostic plots logged as MLflow artifacts.

WHY LOG PLOTS?
--------------
Numbers (roc_auc=0.81) tell you *how good*; plots tell you *how it's good or
bad*. A confusion matrix shows whether you're missing churners (costly) or
crying wolf. An ROC curve shows behaviour across every threshold. Feature
importance shows *why* the model decides -- essential for stakeholder trust and
for debugging data leakage. Storing these in MLflow means every model version
carries its own evidence, forever.

Each function returns a matplotlib Figure so the caller can hand it straight to
`mlflow.log_figure(fig, "plots/name.png")` -- no temp files needed.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend: works in CI / servers with no display

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline


def confusion_matrix_fig(y_true, y_pred) -> plt.Figure:
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["stay", "churn"], cmap="Blues"
    )
    disp.ax_.set_title("Confusion Matrix")
    return disp.figure_


def roc_curve_fig(y_true, y_proba) -> plt.Figure:
    disp = RocCurveDisplay.from_predictions(y_true, y_proba, name="model")
    disp.ax_.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    disp.ax_.set_title("ROC Curve")
    disp.ax_.legend(loc="lower right")
    return disp.figure_


def feature_importance_fig(pipeline: Pipeline, top_n: int = 15) -> plt.Figure:
    """Works for both tree models (feature_importances_) and linear models (coef_)."""
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = np.asarray(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importance (impurity decrease)"
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_)).ravel()
        xlabel = "Importance (|coefficient|)"
    else:  # pragma: no cover - defensive
        raise ValueError("Model exposes neither feature_importances_ nor coef_")

    order = np.argsort(importances)[::-1][:top_n]
    names = feature_names[order][::-1]
    values = importances[order][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, values, color="#2a6f97")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {len(names)} features")
    fig.tight_layout()
    return fig
