"""
Build the preprocessing pipeline (ColumnTransformer).

WHY BUNDLE PREPROCESSING INTO A PIPELINE?
-----------------------------------------
The single most common production ML bug is *training/serving skew*: you scale
and encode features one way during training, then a slightly different way in
the API — and predictions silently go wrong. The fix is to make preprocessing
part of the model object itself. We build one sklearn Pipeline = [preprocess ->
classifier]. When we save it (Phase 1) and load it in the API (Phase 3), the
exact same transformations are applied. No skew, by construction.

- Numeric features: impute missing with the median, then standardize.
- Categorical features: impute with the most frequent value, then one-hot encode.
  `handle_unknown="ignore"` means a category unseen during training won't crash
  prediction at serving time — critical for real traffic.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor
