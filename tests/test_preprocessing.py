"""Tests for the preprocessing pipeline."""
from src.config import FEATURE_COLUMNS
from src.data.generate_dataset import generate_churn_data
from src.features.preprocessing import build_preprocessor


def test_output_row_count_and_expansion():
    """One-hot encoding should EXPAND the number of columns beyond the raw count."""
    df = generate_churn_data(n_rows=200, seed=1)
    pre = build_preprocessor()
    transformed = pre.fit_transform(df[FEATURE_COLUMNS])

    assert transformed.shape[0] == 200
    assert transformed.shape[1] > len(FEATURE_COLUMNS)


def test_handles_unseen_category_without_crashing():
    """A category never seen in training must not crash prediction at serve time."""
    df = generate_churn_data(n_rows=150, seed=2)
    pre = build_preprocessor().fit(df[FEATURE_COLUMNS])

    unseen = df[FEATURE_COLUMNS].head(1).copy()
    unseen.loc[:, "contract_type"] = "Weekly"  # not a real category
    out = pre.transform(unseen)  # should NOT raise

    assert out.shape[0] == 1


def test_numeric_features_are_scaled():
    """After StandardScaler, numeric columns should be roughly zero-mean."""
    df = generate_churn_data(n_rows=500, seed=3)
    pre = build_preprocessor()
    transformed = pre.fit_transform(df[FEATURE_COLUMNS])

    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    n_numeric = 5  # first 5 transformed columns are the scaled numerics
    means = dense[:, :n_numeric].mean(axis=0)
    assert abs(means).max() < 1e-6
