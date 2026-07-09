"""
Central configuration for the whole project.

WHY THIS FILE EXISTS
--------------------
In real MLOps you never scatter magic strings ("data/raw/churn.csv") and
hyperparameters across 10 files. When something changes (a column name, a path,
the random seed) you want ONE place to change it. This module is the single
source of truth for paths, column definitions, and reproducibility settings.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file if present (never fails if missing).
load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# ROOT = the project root (this file lives in <root>/src/config.py).
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / os.getenv("MODEL_DIR", "models")
MLARTIFACTS_DIR = ROOT / "mlartifacts"

RAW_DATA_FILE = RAW_DATA_DIR / "churn.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Make sure the important output folders exist at import time.
for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, MLARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
# NOTE (Phase 2): we moved from the file store (file:./mlruns) to a SQLite
# backend. The Model Registry ONLY works with a database-backed store. SQLite
# is a real database in a single file -- same shape as the PostgreSQL we'll run
# on DigitalOcean later, just zero-config for local dev.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")

# Where run artifacts (models, plots, reports) are stored. With a DB backend we
# must set this explicitly; we use a local folder as a file:// URI.
MLFLOW_ARTIFACT_LOCATION = os.getenv(
    "MLFLOW_ARTIFACT_LOCATION", MLARTIFACTS_DIR.as_uri()
)

# The name under which the best model is registered in the Model Registry.
# The API (Phase 3) will load "the Production version of this name".
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "churn-classifier")

# ---------------------------------------------------------------------------
# Schema: the contract for what our dataset looks like.
# The API (Phase 3) and preprocessing both rely on these exact names.
# ---------------------------------------------------------------------------
TARGET = "churned"

NUMERIC_FEATURES = [
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_calls",
]

CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "contract_type",
    "payment_method",
    "internet_service",
    "tech_support",
    "paperless_billing",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET]
