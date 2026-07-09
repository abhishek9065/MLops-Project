"""
Load the raw dataset and produce a reproducible train/test split.

WHY A DEDICATED MODULE?
-----------------------
The train/test split is a decision, not an afterthought. If different scripts
split differently, your metrics become non-comparable and you can get silent
data leakage. We centralize the split with a FIXED random seed and STRATIFY on
the target so both splits keep the same churn ratio. Every model in this project
is trained and evaluated on the exact same split — that's what makes MLflow
comparisons fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TARGET,
    TEST_SIZE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_raw() -> pd.DataFrame:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_FILE}. "
            f"Run:  python -m src.data.generate_dataset"
        )
    df = pd.read_csv(RAW_DATA_FILE)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], RAW_DATA_FILE)
    return df


def get_split() -> DataSplit:
    df = load_raw()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # keep churn ratio identical in both splits
    )
    logger.info(
        "Split -> train=%d, test=%d (churn rate train=%.3f, test=%.3f)",
        len(X_train),
        len(X_test),
        y_train.mean(),
        y_test.mean(),
    )
    return DataSplit(X_train, X_test, y_train, y_test)
