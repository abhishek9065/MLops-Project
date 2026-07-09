# Verification checklist — "is everything actually working?"

Run these checks **in order** from the `CX Curn Project/` folder. Each step
says the exact command and what you should see. If a step fails, fix it before
moving on — every later step depends on the earlier ones.

> One-time setup first:
>
> ```bash
> python3 -m venv .venv
> source .venv/bin/activate        # Windows: .venv\Scripts\activate
> pip install -r requirements.txt
> ```

---

## 1. Dataset generation (Phase 1)

```bash
make data          # = python -m src.data.generate_dataset
```

**Expect:** `Wrote 7000 rows to .../data/raw/churn.csv` and a churn rate
around 40–43%. The file `data/raw/churn.csv` exists.

## 2. Training + MLflow tracking (Phases 1–2)

```bash
make train         # = python -m src.training.train
```

**Expect:**
- Metric lines for BOTH models, e.g.
  `[logistic_regression] ... roc_auc=0.81` and `[random_forest] ... roc_auc=0.80`
- `Best model: ... (roc_auc=...)`
- `Registered 'churn-classifier' version 1` and `Promoted ... -> Production`
- Files exist: `models/best_model.pkl`, `models/model_metadata.json`, `mlflow.db`

## 3. MLflow UI (Phase 2)

```bash
make mlflow-ui     # then open http://localhost:5000
```

**Expect:** experiment **churn-prediction** with 2 runs; each run has
params, 5 metrics, and artifacts (model + plots). Under **Models**,
`churn-classifier` v1 is in stage **Production**. Ctrl-C to stop.

## 4. Unit tests (Phase 4)

```bash
make test          # = pytest
```

**Expect:** `12 passed`. These same tests run in CI on every push.

## 5. API serving (Phase 3)

Terminal 1:

```bash
make serve         # = uvicorn app.main:app --reload --port 8000
```

Terminal 2 — automated check of every endpoint:

```bash
./scripts/smoke_test.sh
```

**Expect:** `7 passed, 0 failed — SMOKE TEST PASSED`. The script verifies:
health, model-info, a high-risk customer scores **churn**, a low-risk customer
scores **stay**, batch prediction, invalid input is rejected with 422, and
Prometheus metrics are exposed.

Also try it by hand: open http://localhost:8000/docs (interactive Swagger UI)
and execute `POST /predict` with the example payload.

## 6. Docker stack (Phases 5 + 8)

Stop the local API first (Ctrl-C in terminal 1), then:

```bash
docker compose up --build -d
docker compose ps
```

**Expect (after ~1–2 min):** `churn-mlflow` healthy, `churn-trainer`
**Exited (0)** (it's a one-shot job: trains, registers the model, exits),
`churn-api` healthy, `churn-prometheus` and `churn-grafana` up. Then:

```bash
./scripts/smoke_test.sh                 # same 7 checks, now against the container
```

And in the browser:
- http://localhost:8000/docs — API
- http://localhost:5000 — MLflow (model registered by the trainer container)
- http://localhost:9090/targets — Prometheus: `churn-api` target **UP**
- http://localhost:3000 — Grafana (admin/admin): churn dashboard shows
  request rate / latency / predictions after you send a few requests

Tear down with `docker compose down` (add `-v` to also wipe volumes).

## 7. CI/CD (Phase 6)

Push any change to `master` (or merge a PR), then open the repo's
**Actions** tab.

**Expect:** `test` ✅ and `build-and-push` ✅ (image published to GHCR).
The `deploy` job **skips itself with a notice** until you add the
`DROPLET_HOST` / `DROPLET_USER` / `DROPLET_SSH_KEY` secrets — that is
intentional, not a failure. See `docs/github_secrets.md`.

## 8. Drift detection + retraining (Phase 9)

```bash
make simulate-data   # writes data/raw/churn_new.csv with a deliberate shift
make drift
```

**Expect:** `dataset_drift=True`, with `monthly_charges` and
`num_support_calls` flagged **significant** (those are exactly the features
the simulator shifts — the detector finding them proves it works). Exit code
is 1 (that's the "drift found" signal for cron/CI). Reports land in
`reports/drift_report.html` (open it in a browser) and `drift_report.json`.

```bash
make retrain
```

**Expect:** the champion's ROC-AUC on the new data, two challenger scores,
then EITHER `PROMOTED challenger ...` (new registry version in Production)
OR `KEPT champion ...` (challenger didn't beat it by ≥ 0.005). **Both are
correct outcomes** — the point is the gate decides, not you.

```bash
make rollback        # lists versions; add TO=<n> to roll back to version n
```

**Expect:** a list of registered versions with the current Production marked.

---

## Quick pass/fail summary

| # | Check | Command | Pass looks like |
|---|-------|---------|-----------------|
| 1 | Data | `make data` | 7000 rows written |
| 2 | Training | `make train` | 2 runs, model v1 → Production |
| 3 | MLflow | `make mlflow-ui` | runs + registry visible |
| 4 | Tests | `make test` | 12 passed |
| 5 | API | `make serve` + `./scripts/smoke_test.sh` | SMOKE TEST PASSED |
| 6 | Docker | `docker compose up -d` + smoke test | all services up, trainer Exited(0) |
| 7 | CI | push to master | test + build green, deploy skipped |
| 8 | Drift/retrain | `make drift` / `make retrain` | drift found; gate decides promotion |

When all eight pass locally, the project is proven end-to-end and you're
ready for the real deployment: `docs/digitalocean_deploy.md`.
