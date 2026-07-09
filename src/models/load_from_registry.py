"""
Prove the production-loading path works.

Run this AFTER training. It asks the Registry for the current Production model
and makes a prediction -- exactly what the FastAPI app will do in Phase 3.
This is how you validate that "the registry is the source of truth" actually
holds, before wiring it into a web server.

    python -m src.models.load_from_registry
"""
from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLUMNS
from src.models.registry import get_production_version, load_production_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

# One realistic, high-risk customer (month-to-month, fiber, low tenure, many calls).
SAMPLE = {
    "senior_citizen": 0,
    "tenure_months": 2,
    "monthly_charges": 95.5,
    "total_charges": 190.0,
    "num_support_calls": 4,
    "gender": "Female",
    "partner": "No",
    "dependents": "No",
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "tech_support": "No",
    "paperless_billing": "Yes",
}


def main() -> None:
    version = get_production_version()
    if version is None:
        raise SystemExit("No Production model found. Run `python -m src.training.train` first.")

    model = load_production_model()
    X = pd.DataFrame([SAMPLE])[FEATURE_COLUMNS]

    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    logger.info("Production model version: %s", version)
    logger.info("Prediction: %s (churn probability = %.3f)", "CHURN" if pred else "STAY", proba)


if __name__ == "__main__":
    main()
