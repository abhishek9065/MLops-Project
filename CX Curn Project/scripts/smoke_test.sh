#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Smoke test for the Churn Prediction API.
#
# WHAT IT DOES
#   Fires real requests at a RUNNING API and checks every endpoint behaves:
#   health, model-info, predict (high & low risk), batch-predict, input
#   validation, and Prometheus metrics. Exits 0 if all checks pass, 1 if any
#   fail -- so you can use it locally, in CI, or right after a deploy.
#
# USAGE
#   ./scripts/smoke_test.sh                     # against http://localhost:8000
#   ./scripts/smoke_test.sh http://<droplet-ip> # against a deployed instance
#
# Start the API first, either way:
#   make serve            (local venv)
#   docker compose up -d  (full stack)
# ---------------------------------------------------------------------------
set -u

BASE_URL="${1:-http://localhost:8000}"
PASS=0
FAIL=0

check() {
  local name="$1" expected="$2" actual="$3"
  if [[ "$actual" == *"$expected"* ]]; then
    echo "  PASS  $name"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  $name"
    echo "        expected to contain: $expected"
    echo "        got: ${actual:0:200}"
    FAIL=$((FAIL + 1))
  fi
}

echo "Smoke-testing $BASE_URL ..."

# A customer profile that should score HIGH churn risk:
# new customer, month-to-month, fiber, many support calls.
HIGH_RISK='{"senior_citizen":0,"tenure_months":2,"monthly_charges":95.5,"total_charges":190.0,"num_support_calls":4,"gender":"Female","partner":"No","dependents":"No","contract_type":"Month-to-month","payment_method":"Electronic check","internet_service":"Fiber optic","tech_support":"No","paperless_billing":"Yes"}'

# A customer profile that should score LOW churn risk:
# long tenure, two-year contract, no issues.
LOW_RISK='{"senior_citizen":0,"tenure_months":60,"monthly_charges":25.0,"total_charges":1500.0,"num_support_calls":0,"gender":"Male","partner":"Yes","dependents":"Yes","contract_type":"Two year","payment_method":"Credit card","internet_service":"No","tech_support":"Yes","paperless_billing":"No"}'

# 1. Health: service up and a model loaded.
check "GET /health -> healthy" '"status":"healthy"' \
  "$(curl -sf --max-time 10 "$BASE_URL/health")"

# 2. Model info: reports which model/version is serving.
check "GET /model-info -> has model_name" '"model_name"' \
  "$(curl -sf --max-time 10 "$BASE_URL/model-info")"

# 3. High-risk profile should be labelled churn.
check "POST /predict (high risk) -> churn" '"prediction_label":"churn"' \
  "$(curl -sf --max-time 10 -X POST "$BASE_URL/predict" -H 'Content-Type: application/json' -d "$HIGH_RISK")"

# 4. Low-risk profile should be labelled stay.
check "POST /predict (low risk) -> stay" '"prediction_label":"stay"' \
  "$(curl -sf --max-time 10 -X POST "$BASE_URL/predict" -H 'Content-Type: application/json' -d "$LOW_RISK")"

# 5. Batch endpoint returns one prediction per customer.
check "POST /batch-predict (2 customers) -> count 2" '"count":2' \
  "$(curl -sf --max-time 10 -X POST "$BASE_URL/batch-predict" -H 'Content-Type: application/json' -d "{\"customers\":[$HIGH_RISK,$LOW_RISK]}")"

# 6. Validation: garbage in -> 422 with a clear message, NOT a bad prediction.
check "POST /predict (invalid tenure) -> 422 validation error" 'greater_than_equal' \
  "$(curl -s --max-time 10 -X POST "$BASE_URL/predict" -H 'Content-Type: application/json' -d "${HIGH_RISK/\"tenure_months\":2/\"tenure_months\":-5}")"

# 7. Prometheus metrics: prediction counter is exposed and non-zero by now.
check "GET /metrics -> churn_predictions_total exposed" 'churn_predictions_total' \
  "$(curl -sf --max-time 10 "$BASE_URL/metrics")"

echo
echo "Results: $PASS passed, $FAIL failed"
if [[ $FAIL -gt 0 ]]; then
  echo "SMOKE TEST FAILED"
  exit 1
fi
echo "SMOKE TEST PASSED"
