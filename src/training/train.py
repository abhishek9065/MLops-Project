"""
Train candidate models, track EVERYTHING in MLflow, register + promote the best.

PHASE 2 UPGRADES over Phase 1:
  - MLflow backend is now SQLite (enables the Model Registry).
  - Each run logs a model *signature* + *input example* (the model's I/O contract).
  - Each run logs diagnostic plots (confusion matrix, ROC, feature importance)
    and a text classification report as artifacts.
  - The best model is REGISTERED in the Model Registry and promoted to
    'Production', archiving the previous Production version (enables rollback).

The plain best_model.pkl / model_metadata.json are still written for
convenience, but the Registry is now the source of truth for "what's live".
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_FILE,
    FEATURE_COLUMNS,
    MODEL_METADATA_FILE,
    RANDOM_STATE,
    RAW_DATA_FILE,
)
from src.data.load_data import get_split
from src.features.preprocessing import build_preprocessor
from src.models import plots
from src.models.metrics import compute_metrics, pretty
from src.models.registry import register_and_promote, setup_mlflow
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _dataset_fingerprint() -> dict:
    raw_bytes = RAW_DATA_FILE.read_bytes()
    return {"data_md5": hashlib.md5(raw_bytes).hexdigest(), "data_path": str(RAW_DATA_FILE)}


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def train() -> None:
    setup_mlflow()  # sets tracking URI + experiment (SQLite backend)
    split = get_split()
    fingerprint = _dataset_fingerprint()

    results: dict[str, dict] = {}

    for name, estimator in candidate_models().items():
        with mlflow.start_run(run_name=name) as run:
            pipeline = Pipeline(
                steps=[("preprocess", build_preprocessor()), ("model", estimator)]
            )
            pipeline.fit(split.X_train, split.y_train)

            y_pred = pipeline.predict(split.X_test)
            y_proba = pipeline.predict_proba(split.X_test)[:, 1]
            metrics = compute_metrics(split.y_test, y_pred, y_proba)

            # --- params & metrics ---
            mlflow.set_tag("model_family", name)
            mlflow.log_params(estimator.get_params())
            mlflow.log_param("n_features_in", len(FEATURE_COLUMNS))
            mlflow.log_param("n_train_rows", len(split.X_train))
            mlflow.log_param("data_md5", fingerprint["data_md5"])
            mlflow.log_metrics(metrics)

            # --- the model's I/O contract (signature + example input) ---
            signature = infer_signature(split.X_test, y_pred)
            input_example = split.X_test.head(3)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            # --- diagnostic artifacts ---
            mlflow.log_figure(
                plots.confusion_matrix_fig(split.y_test, y_pred),
                "plots/confusion_matrix.png",
            )
            mlflow.log_figure(
                plots.roc_curve_fig(split.y_test, y_proba), "plots/roc_curve.png"
            )
            mlflow.log_figure(
                plots.feature_importance_fig(pipeline), "plots/feature_importance.png"
            )
            report = classification_report(split.y_test, y_pred, target_names=["stay", "churn"])
            mlflow.log_text(report, "reports/classification_report.txt")

            logger.info("[%s] %s", name, pretty(metrics))
            results[name] = {
                "pipeline": pipeline,
                "metrics": metrics,
                "run_id": run.info.run_id,
            }

    # --- select best by ROC-AUC, register + promote ---
    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best = results[best_name]
    logger.info("Best model: %s (roc_auc=%.4f)", best_name, best["metrics"]["roc_auc"])

    version = register_and_promote(best["run_id"], best["metrics"]["roc_auc"])

    # --- also keep local artifacts for convenience ---
    joblib.dump(best["pipeline"], BEST_MODEL_FILE)
    metadata = {
        "model_name": best_name,
        "registered_model_version": version,
        "mlflow_run_id": best["run_id"],
        "metrics": best["metrics"],
        "features": FEATURE_COLUMNS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_md5": fingerprint["data_md5"],
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    logger.info("Saved local copies -> %s , %s", BEST_MODEL_FILE, MODEL_METADATA_FILE)


if __name__ == "__main__":
    train()
