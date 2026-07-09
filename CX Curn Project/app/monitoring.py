"""
Prometheus metrics for the API.

WHAT WE MEASURE AND WHY (the four golden signals, ML-flavoured)
--------------------------------------------------------------
- REQUESTS (traffic): how much load are we under? Sudden drops can mean an
  upstream outage; spikes can mean abuse or a viral feature.
- LATENCY: how long does a prediction take? Users and SLAs care. A creeping p95
  often signals a resource problem before it becomes an outage.
- ERRORS: what fraction of requests fail (5xx)? The single most important
  reliability number.
- PREDICTION DISTRIBUTION (ML-specific): what fraction of predictions are
  "churn"? If this suddenly jumps from 40% to 90%, your MODEL or your INPUT DATA
  changed -- an early smoke alarm for data drift (Phase 9), visible in real time.

Prometheus SCRAPES the /metrics endpoint every few seconds and stores the
numbers as time series; Grafana graphs them.
"""
from __future__ import annotations

import time

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# --- HTTP-level metrics (recorded for every request by the middleware) ---
REQUEST_COUNT = Counter(
    "churn_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "churn_http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["path"],
)

# --- Prediction-level metrics (recorded by the inference endpoints) ---
PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total individual customer predictions made",
)
PREDICTIONS_BY_LABEL = Counter(
    "churn_predictions_by_label_total",
    "Predictions broken down by predicted label",
    ["label"],  # "churn" or "stay"
)
PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total",
    "Errors raised while producing predictions",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Times every request and records count + latency, labelled by route."""

    async def dispatch(self, request: Request, call_next):
        # Use the route template (e.g. "/predict") not the raw URL, so metrics
        # don't explode into thousands of unique label values.
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.scope.get("route").path if request.scope.get("route") else request.url.path
        REQUEST_LATENCY.labels(path=path).observe(elapsed)
        REQUEST_COUNT.labels(
            method=request.method, path=path, status=str(response.status_code)
        ).inc()
        return response


def record_predictions(labels) -> None:
    """Increment prediction counters given an iterable of 0/1 labels."""
    for label in labels:
        PREDICTIONS_TOTAL.inc()
        PREDICTIONS_BY_LABEL.labels(label="churn" if int(label) == 1 else "stay").inc()
