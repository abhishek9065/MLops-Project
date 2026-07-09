"""
Metric computation, in one place.

WHY ROC-AUC AND F1 MATTER FOR CHURN
-----------------------------------
Churn is imbalanced (most customers don't churn). Plain accuracy is misleading:
a model that predicts "nobody churns" can be 80%+ accurate and utterly useless.
So we track precision (of those we flagged, how many really churned), recall
(of the churners, how many we caught), F1 (their balance), and ROC-AUC (ranking
quality independent of threshold). We select the best model on ROC-AUC.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def pretty(metrics: Dict[str, float]) -> str:
    return " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
