"""
Shared pytest fixtures.

KEY IDEA: tests must run on a FRESH checkout (like CI) where no trained model
exists. So `ensure_model` trains a tiny model to disk if one isn't already
there. It does NOT clobber an existing model, so running tests locally reuses
whatever you trained. This keeps tests fast (500 rows, no MLflow overhead) and
hermetic (no network, no tracking server required).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import BEST_MODEL_FILE, FEATURE_COLUMNS, MODEL_METADATA_FILE, TARGET
from src.data.generate_dataset import generate_churn_data
from src.features.preprocessing import build_preprocessor


@pytest.fixture(scope="session")
def ensure_model():
    """Guarantee a local model file exists so the API's file-fallback works."""
    if not BEST_MODEL_FILE.exists():
        df = generate_churn_data(n_rows=500, seed=42)
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
            ]
        )
        pipeline.fit(df[FEATURE_COLUMNS], df[TARGET])
        joblib.dump(pipeline, BEST_MODEL_FILE)
        MODEL_METADATA_FILE.write_text(
            json.dumps(
                {
                    "model_name": "test_logreg",
                    "registered_model_version": "test",
                    "metrics": {"roc_auc": 0.8},
                    "features": FEATURE_COLUMNS,
                    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
    return BEST_MODEL_FILE


@pytest.fixture
def sample_customer() -> dict:
    return {
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


@pytest.fixture
def client(ensure_model):
    """FastAPI TestClient. The `with` block runs startup (model load) + shutdown."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
