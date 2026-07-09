"""
Thin wrapper around MLflow tracking + Model Registry.

WHY A REGISTRY LAYER?
---------------------
The rest of the app should never hardcode run IDs or file paths. It should speak
in intentions: "register this model", "promote to Production", "load whatever is
in Production". This module is that vocabulary. When we later swap SQLite for a
remote PostgreSQL-backed server on DigitalOcean, NONE of the calling code
changes -- only MLFLOW_TRACKING_URI in the environment.

A note on stages vs aliases:
  MLflow historically uses *stages* (None/Staging/Production/Archived). Newer
  MLflow is moving toward *aliases* (e.g. @champion). We use stages here because
  they're the most widely recognized concept and map cleanly to "what's live".
  The alias equivalent is shown in comments.
"""
from __future__ import annotations

from datetime import datetime, timezone

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRODUCTION_STAGE = "Production"


def setup_mlflow() -> MlflowClient:
    """Point MLflow at our backend and ensure the experiment exists.

    Creating the experiment with an explicit artifact_location is required when
    using a database backend -- otherwise MLflow doesn't know where to put files.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        # With a REMOTE server (http://...) we let the server manage the artifact
        # store (it uses `--serve-artifacts`). With a LOCAL db backend we must
        # point the artifact store at a local folder ourselves.
        if MLFLOW_TRACKING_URI.startswith("http"):
            client.create_experiment(MLFLOW_EXPERIMENT_NAME)
        else:
            client.create_experiment(
                MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION
            )
        logger.info("Created experiment '%s'", MLFLOW_EXPERIMENT_NAME)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return client


def register_and_promote(run_id: str, roc_auc: float) -> str:
    """Register the model from a run and promote it to Production.

    Returns the new model version string. `archive_existing_versions=True` means
    the previously-live model is moved to 'Archived' -- that archived version is
    exactly what we'd roll back to (Phase 9).
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    logger.info(
        "Registered '%s' version %s from run %s",
        REGISTERED_MODEL_NAME,
        version.version,
        run_id,
    )

    # Describe *why* this version exists -- future-you will thank present-you.
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        description=f"Auto-registered. Test ROC-AUC={roc_auc:.4f}.",
    )

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version.version,
        stage=PRODUCTION_STAGE,
        archive_existing_versions=True,
    )
    # Alias equivalent (modern MLflow):
    #   client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    logger.info(
        "Promoted '%s' v%s -> %s (previous Production archived)",
        REGISTERED_MODEL_NAME,
        version.version,
        PRODUCTION_STAGE,
    )
    return version.version


def load_production_model():
    """Load whatever model is currently in Production. Used by the API (Phase 3)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{PRODUCTION_STAGE}"
    logger.info("Loading model from %s", model_uri)
    return mlflow.sklearn.load_model(model_uri)


def get_production_version() -> str | None:
    """Return the version number currently in Production, or None."""
    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[PRODUCTION_STAGE])
    return versions[0].version if versions else None


def list_versions() -> list[dict]:
    """All versions of the registered model with their current stage."""
    client = MlflowClient()
    out = []
    for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"):
        out.append({"version": v.version, "stage": v.current_stage, "run_id": v.run_id})
    return sorted(out, key=lambda d: int(d["version"]))


def rollback_to(version: str) -> None:
    """Promote a specific (usually archived) version back to Production.

    ROLLBACK = putting a previously-good version back in charge. Because our
    promotion archives the prior Production version instead of deleting it, that
    exact model is still here and can be reinstated in seconds -- no retraining,
    no rebuild. This is the safety net for a bad deploy.
    """
    client = MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage=PRODUCTION_STAGE,
        archive_existing_versions=True,
    )
    logger.info("Rolled back: '%s' v%s is now Production", REGISTERED_MODEL_NAME, version)


def get_production_details() -> dict | None:
    """Fetch version + originating run info for the Production model.

    Lets the API report an accurate /model-info (name, metrics, trained-at) even
    inside a container that has no local model_metadata.json -- the truth comes
    from the tracking server, which is exactly where it should come from.
    """
    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[PRODUCTION_STAGE])
    if not versions:
        return None
    version = versions[0]
    run = client.get_run(version.run_id)
    wanted = ("accuracy", "precision", "recall", "f1", "roc_auc")
    trained_at = None
    if run.info.start_time:
        trained_at = datetime.fromtimestamp(
            run.info.start_time / 1000, tz=timezone.utc
        ).isoformat()
    return {
        "version": str(version.version),
        "run_id": version.run_id,
        "model_name": run.data.tags.get("model_family", "unknown"),
        "metrics": {k: run.data.metrics[k] for k in wanted if k in run.data.metrics},
        "trained_at": trained_at,
    }
