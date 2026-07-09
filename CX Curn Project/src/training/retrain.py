"""
Retrain on new data and promote the challenger ONLY if it beats the champion.

THE CORE MLOps IDEA: champion vs challenger
-------------------------------------------
- CHAMPION  = the model currently in Production.
- CHALLENGER = a fresh model trained on the new data.

The honest way to compare them is on the SAME held-out slice of the NEW data
(the world the model must operate in now). We:
  1. Split the new data into train/test.
  2. Evaluate the current champion on the new TEST set -> baseline ROC-AUC.
  3. Train challengers on the new TRAIN set, pick the best on the new TEST set.
  4. Promote the challenger ONLY if it beats the champion by a margin.

Why the margin (MIN_IMPROVEMENT)? To avoid churning Production for noise. A
0.001 AUC bump isn't worth the risk of a swap; we require a real improvement.

If the challenger doesn't win, we keep the champion. That IS the safe default --
and combined with archived versions, rollback stays trivial.
"""
from __future__ import annotations

import argparse

import mlflow
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, RANDOM_STATE, TARGET, TEST_SIZE
from src.data.load_data import DataSplit
from src.features.preprocessing import build_preprocessor
from src.models.metrics import compute_metrics, pretty
from src.models.registry import register_and_promote, setup_mlflow
from src.training.train import candidate_models
from src.utils.logger import get_logger

import pandas as pd
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

MIN_IMPROVEMENT = 0.005  # challenger must beat champion by at least this ROC-AUC


def _split(df: pd.DataFrame) -> DataSplit:
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return DataSplit(X_train, X_test, y_train, y_test)


def _champion_auc_on(X_test, y_test) -> float | None:
    """Evaluate the current Production model on the NEW test set."""
    try:
        from src.models.registry import load_production_model

        champion = load_production_model()
        proba = champion.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = compute_metrics(y_test, preds, proba)["roc_auc"]
        logger.info("Champion ROC-AUC on new data: %.4f", auc)
        return auc
    except Exception as exc:  # noqa: BLE001
        logger.warning("No champion to compare against (%s). Any challenger wins.", exc)
        return None


def retrain(new_data_path: str) -> None:
    setup_mlflow()
    df = pd.read_csv(new_data_path)
    logger.info("Loaded %d rows of new data from %s", len(df), new_data_path)
    split = _split(df)

    champion_auc = _champion_auc_on(split.X_test, split.y_test)

    results: dict[str, dict] = {}
    for name, estimator in candidate_models().items():
        with mlflow.start_run(run_name=f"retrain_{name}") as run:
            pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
            pipeline.fit(split.X_train, split.y_train)
            proba = pipeline.predict_proba(split.X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)
            metrics = compute_metrics(split.y_test, preds, proba)

            mlflow.set_tag("stage", "retrain")
            mlflow.set_tag("model_family", name)
            mlflow.log_params(estimator.get_params())
            mlflow.log_metrics(metrics)
            if champion_auc is not None:
                mlflow.log_metric("champion_roc_auc", champion_auc)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            logger.info("[challenger %s] %s", name, pretty(metrics))
            results[name] = {"metrics": metrics, "run_id": run.info.run_id}

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best = results[best_name]
    challenger_auc = best["metrics"]["roc_auc"]

    baseline = champion_auc if champion_auc is not None else float("-inf")
    if challenger_auc > baseline + MIN_IMPROVEMENT:
        version = register_and_promote(best["run_id"], challenger_auc)
        logger.info(
            "PROMOTED challenger '%s' (AUC %.4f > champion %.4f + %.3f). New Production v%s.",
            best_name,
            challenger_auc,
            baseline if baseline != float("-inf") else 0.0,
            MIN_IMPROVEMENT,
            version,
        )
    else:
        logger.info(
            "KEPT champion. Challenger '%s' AUC %.4f did not beat champion %.4f by >= %.3f.",
            best_name,
            challenger_auc,
            baseline,
            MIN_IMPROVEMENT,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain and conditionally promote.")
    parser.add_argument("--data", required=True, help="Path to the new training data CSV")
    args = parser.parse_args()
    retrain(args.data)


if __name__ == "__main__":
    main()
