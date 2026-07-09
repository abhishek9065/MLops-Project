"""Tests for the model loading + prediction service."""
from src.config import FEATURE_COLUMNS
from src.data.generate_dataset import generate_churn_data


def test_service_loads_and_is_ready(ensure_model):
    from app.model_loader import ModelService

    service = ModelService()
    service.load()
    assert service.is_ready


def test_predict_frame_shapes_and_ranges(ensure_model):
    from app.model_loader import ModelService

    service = ModelService()
    service.load()

    df = generate_churn_data(n_rows=10, seed=7)[FEATURE_COLUMNS]
    labels, proba = service.predict_frame(df)

    assert len(labels) == 10
    assert len(proba) == 10
    assert all(0.0 <= p <= 1.0 for p in proba)
    assert set(int(x) for x in labels).issubset({0, 1})
