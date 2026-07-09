"""
FastAPI application for churn prediction.

Endpoints
---------
GET  /health         liveness/readiness -- is the service up and is a model loaded?
GET  /model-info     which model is serving, its version, source, and metrics
POST /predict        score a single customer
POST /batch-predict  score up to 1000 customers in one call

Run locally:
    uvicorn app.main:app --reload --port 8000
    open http://localhost:8000/docs   (interactive Swagger UI)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.requests import Request

from app.model_loader import model_service
from app.monitoring import (
    PREDICTION_ERRORS,
    PrometheusMiddleware,
    record_predictions,
)
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, when the app boots (not per request)."""
    logger.info("Starting up: loading model...")
    try:
        model_service.load()
    except Exception as exc:  # noqa: BLE001
        # We log but don't crash: /health will report model_loaded=False so an
        # orchestrator can see the service is degraded instead of crash-looping.
        logger.error("Model failed to load at startup: %s", exc)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Serves the Production churn model from the MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)

# Records traffic, latency and error metrics for every request.
app.add_middleware(PrometheusMiddleware)


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    """Prometheus scrape target. Returns metrics in the text exposition format."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _to_response(label: int, proba: float) -> PredictionResponse:
    return PredictionResponse(
        prediction=int(label),
        prediction_label="churn" if label == 1 else "stay",
        churn_probability=round(float(proba), 4),
        model_name=model_service.model_name,
        model_version=model_service.model_version,
    )


def _require_model() -> None:
    if not model_service.is_ready:
        # 503 = service temporarily unable to handle the request.
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if model_service.is_ready else "degraded",
        model_loaded=model_service.is_ready,
        model_version=model_service.model_version if model_service.is_ready else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    _require_model()
    return ModelInfoResponse(
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        model_source=model_service.source,
        metrics=model_service.metrics,
        features=model_service.features,
        trained_at=model_service.trained_at,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(customer: CustomerFeatures) -> PredictionResponse:
    _require_model()
    try:
        df = pd.DataFrame([customer.model_dump()])
        labels, proba = model_service.predict_frame(df)
        record_predictions(labels)
        return _to_response(labels[0], proba[0])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["inference"])
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    _require_model()
    try:
        df = pd.DataFrame([c.model_dump() for c in request.customers])
        labels, proba = model_service.predict_frame(df)
        record_predictions(labels)
        preds = [_to_response(l, p) for l, p in zip(labels, proba)]
        return BatchPredictResponse(predictions=preds, count=len(preds))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        PREDICTION_ERRORS.inc()
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last-resort handler so clients always get JSON, never a raw stack trace."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
