"""Tests for the FastAPI endpoints (via TestClient, no real server needed)."""


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["status"] == "healthy"


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert "features" in body and len(body["features"]) == 13
    assert "model_version" in body


def test_predict_single(client, sample_customer):
    resp = client.post("/predict", json=sample_customer)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in (0, 1)
    assert body["prediction_label"] in ("churn", "stay")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert "model_version" in body


def test_batch_predict(client, sample_customer):
    resp = client.post("/batch-predict", json={"customers": [sample_customer, sample_customer]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2


def test_invalid_category_rejected(client, sample_customer):
    bad = {**sample_customer, "internet_service": "Fibre optic"}  # typo -> invalid
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_negative_number_rejected(client, sample_customer):
    bad = {**sample_customer, "tenure_months": -5}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_missing_field_rejected(client, sample_customer):
    bad = dict(sample_customer)
    del bad["gender"]
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
