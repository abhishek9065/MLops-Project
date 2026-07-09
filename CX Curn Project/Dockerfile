# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Production image for the Churn Prediction API.
# Same base Python (3.12) we validated locally -> no "works on my machine".
# ---------------------------------------------------------------------------
FROM python:3.12-slim

# PYTHONDONTWRITEBYTECODE: don't litter .pyc files.
# PYTHONUNBUFFERED: stream logs immediately (critical for container logging).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libgomp1: OpenMP runtime required by scikit-learn / xgboost. Without it the
# app imports fine but CRASHES at predict time. curl: used by HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements FIRST and install. Because this layer only changes when
# requirements.txt changes, Docker caches the (slow) pip install across code
# edits -- rebuilds after a code change take seconds, not minutes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Now copy the application code.
COPY src ./src
COPY app ./app
COPY pyproject.toml ./

# Run as a non-root user (security best practice).
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Docker marks the container "healthy" only once /health responds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Uvicorn serves the FastAPI app. host 0.0.0.0 so it's reachable from outside
# the container. No --reload in production.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
