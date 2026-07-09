# Retraining & rollback

## The problem

Models decay. The world the model was trained on (prices, customer mix, behavior)
drifts away from the world it now operates in, and accuracy silently drops. You
need a loop that (a) notices, (b) tries a fresh model, and (c) only swaps if the
new one is genuinely better — with a fast undo if a swap goes wrong.

## The loop in this project

```
simulate ──> drift ──> retrain (champion vs challenger) ──> promote? ──> rollback (if needed)
```

### 1. Detect (`src/monitoring/drift.py`)

PSI + KS compare training data vs recent data. Output: `reports/drift_report.*`
and a non-zero exit code when dataset drift is detected (so a cron/CI job can
trigger the next step automatically).

### 2. Retrain — champion vs challenger (`src/training/retrain.py`)

- **Champion** = current Production model. **Challenger** = model trained on the
  new data.
- Both are scored on the **same held-out slice of the new data** — the only fair
  comparison, because that's the distribution the model must handle now.
- The challenger is promoted **only if** its ROC-AUC beats the champion by at
  least `MIN_IMPROVEMENT = 0.005`. This margin prevents churning Production for
  statistical noise.
- If the challenger doesn't win, we **keep the champion**. Safe default.

> Example from a real run in this repo: champion scored 0.770 on new data, best
> challenger 0.768 → **kept the champion**. The guardrail did its job.

### 3. Promote (`src/models/registry.py :: register_and_promote`)

Registers the challenger and transitions it to `Production` with
`archive_existing_versions=True` — the old Production version becomes `Archived`,
not deleted.

### 4. Rollback (`src/models/registry.py :: rollback_to`, `src/models/rollback.py`)

Because old versions are archived (not gone), rollback is just re-promoting one:

```bash
python -m src.models.rollback          # see all versions + stages
python -m src.models.rollback --to 3   # version 3 becomes Production again
```

The API picks up the change on its next restart (it loads
`models:/churn-classifier/Production`). No retrain, no rebuild.

## When to roll back

- A newly promoted model shows worse business metrics in production.
- A data/feature bug slipped into the last training run.
- Latency or errors spike after a deploy.

Roll back first (restore service), then investigate — the classic incident
response order.

## Modern MLflow note

We use **stages** (`Production`/`Archived`) for clarity. Newer MLflow favors
**aliases** (e.g. `@champion`); the same archive-don't-delete principle applies.
The alias equivalent is shown in comments in `registry.py`.
