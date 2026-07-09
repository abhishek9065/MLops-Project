"""
Loads and holds the model for the API's lifetime.

DESIGN: registry-first, file-fallback.
--------------------------------------
1. Preferred: load `models:/churn-classifier/Production` from MLflow. This means
   whenever you promote/rollback a version in the registry, a simple API restart
   picks it up -- no rebuild, no code change.
2. Fallback: if MLflow is unreachable (e.g. tracking server down at boot), load
   the local models/best_model.pkl so the service still starts and serves.

We load ONCE at startup and keep the object in memory. Reloading per request
would add hundreds of ms of latency to every call.

`model_metadata.json` gives us the human-facing name/metrics/version to report
via /model-info, regardless of which source the model came from.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from threading import Lock

import joblib
import pandas as pd

from src.config import BEST_MODEL_FILE, FEATURE_COLUMNS, MODEL_METADATA_FILE
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHURN_THRESHOLD = 0.5


class ModelService:
    """Singleton-style holder for the loaded model + its metadata."""

    def __init__(self) -> None:
        self.model = None
        self.model_name: str = "unknown"
        self.model_version: str = "unknown"
        self.source: str = "none"
        self.metrics: dict = {}
        self.trained_at: str | None = None
        self.features = FEATURE_COLUMNS
        self._lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _read_metadata(self) -> None:
        if MODEL_METADATA_FILE.exists():
            meta = json.loads(MODEL_METADATA_FILE.read_text())
            self.model_name = meta.get("model_name", self.model_name)
            self.model_version = str(meta.get("registered_model_version", self.model_version))
            self.metrics = meta.get("metrics", {})
            self.trained_at = meta.get("trained_at_utc")

    def load(self) -> None:
        """Load the model. Called once from the app lifespan hook."""
        with self._lock:
            self._read_metadata()

            # 1) Try the MLflow Model Registry (Production stage).
            try:
                from src.models.registry import get_production_details, load_production_model

                self.model = load_production_model()
                details = get_production_details()
                if details:
                    self.model_version = details["version"]
                    self.model_name = details["model_name"]
                    if details["metrics"]:
                        self.metrics = details["metrics"]
                    if details["trained_at"]:
                        self.trained_at = details["trained_at"]
                self.source = "mlflow_registry"
                logger.info("Model loaded from MLflow registry (v%s)", self.model_version)
                return
            except Exception as exc:  # noqa: BLE001 - we intentionally fall back
                logger.warning("Registry load failed (%s). Falling back to local file.", exc)

            # 2) Fall back to the local pickle.
            if not BEST_MODEL_FILE.exists():
                raise RuntimeError(
                    f"No model available: registry failed and {BEST_MODEL_FILE} is missing. "
                    f"Run `python -m src.training.train` first."
                )
            self.model = joblib.load(BEST_MODEL_FILE)
            self.source = "local_file"
            logger.info("Model loaded from local file %s", BEST_MODEL_FILE)

    def predict_frame(self, df: pd.DataFrame):
        """Return (labels, probabilities) for a DataFrame of features."""
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")
        # Enforce column order; the pipeline handles preprocessing internally.
        X = df[self.features]
        proba = self.model.predict_proba(X)[:, 1]
        labels = (proba >= CHURN_THRESHOLD).astype(int)
        return labels, proba

    def loaded_at(self) -> str:
        return datetime.now(timezone.utc).isoformat()


# Single shared instance imported by the app.
model_service = ModelService()
