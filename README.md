# End-to-End MLOps Pipeline for Customer Churn Prediction

![Python](https://img.shields.io/badge/python-3.12-blue)
![Tests](https://img.shields.io/badge/tests-12%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2)
![Docker](https://img.shields.io/badge/containerized-Docker-2496ED)

A **production-style** machine learning system that predicts customer churn and
demonstrates the full MLOps lifecycle: reproducible training, experiment tracking,
a model registry, a REST API, containerization, CI/CD, cloud deployment,
monitoring, and automated drift detection & retraining.

Built as a portfolio project to show not just *a model*, but the **engineering
around a model** that real teams care about.

---

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Local development](#local-development)
- [Docker usage](#docker-usage)
- [API documentation](#api-documentation)
- [MLflow usage](#mlflow-usage)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Deployment (DigitalOcean)](#deployment-digitalocean)
- [Monitoring](#monitoring)
- [Drift detection & retraining](#drift-detection--retraining)
- [Screenshots](#screenshots)
- [Future improvements](#future-improvements)
- [License](#license)

---

## Overview

**Problem:** Losing customers (churn) is expensive; catching at-risk customers
early lets a business intervene. **Solution:** a binary classifier that scores a
customer's churn probability, wrapped in a system that can be trained, deployed,
monitored, and safely updated over time.

**What makes this "MLOps" and not just a notebook:**

- Training is **reproducible** (fixed seeds, versioned data hash) and **tracked**
  (every run logged to MLflow with params, metrics, plots, and a model signature).
- The serving model is chosen via a **Model Registry** (`Production` stage), so
  promotion and **rollback** are first-class operations.
- The API bundles preprocessing *with* the model to eliminate **train/serve skew**.
- Everything is **containerized**, **tested in CI**, and **deployable** with one
  push. Live **monitoring** and a **drift → retrain** loop keep it healthy.

---

## Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │                  TRAINING                     │
   data/raw/churn.csv ──▶│  generate → split → preprocess → train (LR/RF)│
   (synthetic, seeded)   │  compare on ROC-AUC → pick best               │
                         └───────────────┬───────────────────────────────┘
                                         │ log params/metrics/plots/signature
                                         ▼
                         ┌───────────────────────────────────────────────┐
                         │              MLflow Tracking + Registry        │
                         │   experiments  +  churn-classifier @Production │
                         └───────────────┬───────────────────────────────┘
                                         │ load models:/churn-classifier/Production
                                         ▼
        clients ───HTTP──▶ ┌──────────────────────────────┐   /metrics   ┌────────────┐
                           │        FastAPI service        │ ───────────▶ │ Prometheus │
   /health /model-info     │  (Pydantic validation +       │              └─────┬──────┘
   /predict /batch-predict │   registry-first model load)  │                    ▼
                           └──────────────────────────────┘              ┌────────────┐
                                                                          │  Grafana   │
                                                                          └────────────┘
  ── Feedback loop ─────────────────────────────────────────────────────────────────
   new data ─▶ drift (PSI + KS) ─▶ retrain (champion vs challenger) ─▶ promote if better
                                                                    └▶ rollback if needed

  ── Delivery ──────────────────────────────────────────────────────────────────────
   git push ─▶ GitHub Actions: test ─▶ build image ─▶ push GHCR ─▶ SSH deploy to Droplet
```

---

## Tech stack

| Layer               | Tools                                             |
| ------------------- | ------------------------------------------------- |
| Language            | Python 3.12                                       |
| ML                  | scikit-learn, XGBoost, pandas, NumPy              |
| Experiment tracking | MLflow (tracking + Model Registry, SQLite/Postgres) |
| Serving             | FastAPI, Uvicorn, Pydantic v2                     |
| Testing             | pytest, httpx (TestClient)                        |
| Packaging           | Docker, docker-compose                            |
| CI/CD               | GitHub Actions, GitHub Container Registry (GHCR)  |
| Cloud               | DigitalOcean Droplet, Nginx, Certbot (HTTPS)      |
| Monitoring          | Prometheus, Grafana                               |
| Drift detection     | PSI + Kolmogorov–Smirnov (SciPy)                  |

---

## Repository structure

```
MLops/
├── app/                     # FastAPI service
│   ├── main.py              #   endpoints + lifespan model load + /metrics
│   ├── schemas.py           #   Pydantic request/response contracts
│   ├── model_loader.py      #   registry-first, file-fallback loader
│   └── monitoring.py        #   Prometheus metrics + middleware
├── src/
│   ├── config.py            # single source of truth: paths, schema, seeds, MLflow
│   ├── data/                # generate_dataset.py, load_data.py
│   ├── features/            # preprocessing.py (ColumnTransformer)
│   ├── models/              # metrics, plots, registry, load_from_registry, rollback
│   ├── monitoring/          # drift.py (PSI+KS), simulate.py
│   ├── training/            # train.py, retrain.py
│   └── utils/               # logger.py
├── tests/                   # pytest: preprocessing, model loading, API
├── notebooks/               # 01_exploratory_data_analysis.ipynb
├── monitoring/              # prometheus.yml + grafana provisioning & dashboard
├── deploy/                  # setup_droplet.sh, nginx/churn-api.conf
├── docs/                    # deployment, secrets, retraining, screenshots
├── .github/workflows/       # ci-cd.yml
├── Dockerfile
├── docker-compose.yml       # local: mlflow + trainer + api + prometheus + grafana
├── docker-compose.prod.yml  # droplet: pulls prebuilt GHCR image
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .env.example / .env.prod.example
└── README.md
```

---

## Quickstart

```bash
# 1. Environment (Python 3.12 recommended — the ML stack lags newer Pythons)
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash;  ./bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env

# 2. Data + train (logs to MLflow, registers best model to Production)
python -m src.data.generate_dataset
python -m src.training.train

# 3. Serve
uvicorn app.main:app --reload --port 8000     # docs at http://localhost:8000/docs
```

> Prefer one command? The `Makefile` wraps these: `make data`, `make train`,
> `make serve`, `make test`, `make mlflow-ui`, `make compose-up`, `make drift`, …

---

## Local development

- **Config lives in `src/config.py`** — paths, the feature schema, the random
  seed, and MLflow settings. Change things there, not in scattered strings.
- **Reproducibility:** dataset generation and the train/test split are seeded;
  the training run logs an MD5 of the data so a model can always be traced to its
  exact inputs.
- **No train/serve skew:** preprocessing is part of the sklearn `Pipeline`, saved
  and loaded as one object.

---

## Docker usage

One command brings up the whole stack (MLflow → one-shot trainer → API →
Prometheus → Grafana), with startup order enforced by health/completion gates:

```bash
docker compose up --build
#   API docs    → http://localhost:8000/docs
#   MLflow UI   → http://localhost:5000
#   Prometheus  → http://localhost:9090
#   Grafana     → http://localhost:3000   (admin / admin)

docker compose logs -f api
docker compose down -v        # stop + wipe volumes
```

The `Dockerfile` uses `python:3.12-slim`, installs `libgomp1` (required by
scikit-learn/XGBoost at runtime), caches the pip layer, and runs as a **non-root**
user with a `HEALTHCHECK` on `/health`.

---

## API documentation

Interactive Swagger UI is auto-generated at **`/docs`**.

| Method & path         | Description                                             |
| --------------------- | ------------------------------------------------------ |
| `GET  /health`        | Liveness/readiness; reports whether a model is loaded. |
| `GET  /model-info`    | Model name, version, source (registry/file), metrics.  |
| `GET  /metrics`       | Prometheus metrics (scrape target).                    |
| `POST /predict`       | Score one customer.                                    |
| `POST /batch-predict` | Score up to 1000 customers in one request.             |

**Example**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "senior_citizen":0,"tenure_months":2,"monthly_charges":95.5,"total_charges":190.0,
  "num_support_calls":4,"gender":"Female","partner":"No","dependents":"No",
  "contract_type":"Month-to-month","payment_method":"Electronic check",
  "internet_service":"Fiber optic","tech_support":"No","paperless_billing":"Yes"}'
# → {"prediction":1,"prediction_label":"churn","churn_probability":0.9725,
#    "model_name":"logistic_regression","model_version":"1"}
```

Invalid input (bad category, negative number, missing field) returns **422** with
a precise message — Pydantic validates before anything reaches the model.

---

## MLflow usage

- **Backend:** SQLite locally (`mlflow.db`) — required for the Model Registry;
  PostgreSQL in production.
- **Logged per run:** params, metrics (accuracy/precision/recall/F1/ROC-AUC),
  a **model signature** + input example, and diagnostic **plots** (confusion
  matrix, ROC curve, feature importance) + a classification report.
- **Registry:** the best model is registered as `churn-classifier` and promoted
  to `Production`; the API loads `models:/churn-classifier/Production`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Remote tracking server on DigitalOcean: see
[`docs/mlflow_on_digitalocean.md`](docs/mlflow_on_digitalocean.md).

---

## Testing

```bash
pytest        # 12 tests: preprocessing, model loading, API endpoints
```

Tests are **hermetic** — if no trained model exists (fresh CI checkout), a tiny
model is trained to disk automatically. No network or MLflow server needed.

---

## CI/CD

`.github/workflows/ci-cd.yml`, on every push/PR to `main`:

1. **test** — install deps + `pytest` (the quality gate).
2. **build-and-push** *(push to main)* — build image, push to GHCR (SHA + `latest`).
3. **deploy** *(push to main)* — SSH to the droplet, `docker compose pull && up -d`.

Set secrets safely per [`docs/github_secrets.md`](docs/github_secrets.md):
`DROPLET_HOST`, `DROPLET_USER`, `DROPLET_SSH_KEY`, `GHCR_USER`, `GHCR_PAT`.

---

## Deployment (DigitalOcean)

Budget-friendly and **GPU-free** — a 2 GB / 2 vCPU droplet (~$12–18/mo) runs the
whole stack. Full step-by-step (create droplet → provision → firewall → run →
Nginx → HTTPS with Certbot) in
[`docs/digitalocean_deploy.md`](docs/digitalocean_deploy.md). Helpers:
`deploy/setup_droplet.sh`, `deploy/nginx/churn-api.conf`,
`docker-compose.prod.yml`.

---

## Monitoring

The API exposes Prometheus metrics at `/metrics`; Grafana ships a provisioned
**"Churn API Monitoring"** dashboard.

| Metric | Why it matters |
| ------ | -------------- |
| `churn_http_requests_total` | Traffic + error rate (by route & status code). |
| `churn_http_request_latency_seconds` | Latency; a rising p95 warns of trouble early. |
| `churn_predictions_total` | Prediction volume. |
| `churn_predictions_by_label_total` | Churn-vs-stay share — a sudden jump hints at **data drift**, live. |
| `churn_prediction_errors_total` | Inference failures. |

---

## Drift detection & retraining

```bash
python -m src.monitoring.simulate                                # make drifted "new" data
python -m src.monitoring.drift --current data/raw/churn_new.csv  # PSI + KS report
python -m src.training.retrain --data data/raw/churn_new.csv     # promote only if better
python -m src.models.rollback --to 1                             # instant rollback
```

- **Drift** = PSI (≥ 0.25 significant) + KS test → `reports/drift_report.{json,html}`.
- **Retraining** is **champion vs challenger**: both scored on the *new* holdout;
  the challenger wins only if it beats the champion by ≥ 0.005 ROC-AUC — otherwise
  the champion stays. Retraining never blindly replaces.
- **Rollback** is instant because promotion *archives* (never deletes) old
  versions. Details in
  [`docs/retraining_and_rollback.md`](docs/retraining_and_rollback.md).

---

## Screenshots

_Add these images to `docs/screenshots/` and they'll render here:_

| MLflow experiments | API Swagger docs | Grafana dashboard |
| ------------------ | ---------------- | ----------------- |
| `docs/screenshots/mlflow.png` | `docs/screenshots/api_docs.png` | `docs/screenshots/grafana.png` |

---

## Future improvements

- Swap SQLite → **PostgreSQL** and local artifacts → **DigitalOcean Spaces (S3)**.
- **Feature store** + **DVC** for true data versioning.
- **Authentication** (API keys / OAuth) and per-key rate limiting.
- **Shadow deployment / A-B testing** before promoting a challenger.
- **Alerting** (Prometheus Alertmanager) on error-rate, latency, and drift.
- Ground-truth **feedback ingestion** to measure live model performance, not just drift.
- Move MLflow stages → **aliases** (modern MLflow) and add model **lineage** tags.

---

## License

[MIT](LICENSE) © 2026 Abhishek
