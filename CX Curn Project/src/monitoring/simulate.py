"""
Simulate NEW incoming data that has drifted from the training distribution.

This is a stand-in for "real production traffic collected over the last month".
We deliberately shift several distributions so drift detection has something to
find, and so retraining has a reason to produce a different model:
  - prices went up (monthly_charges shifted higher)
  - more customers on month-to-month contracts
  - customers contacting support more often
"""
from __future__ import annotations

import numpy as np

from src.config import RAW_DATA_DIR
from src.data.generate_dataset import generate_churn_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

NEW_DATA_FILE = RAW_DATA_DIR / "churn_new.csv"


def simulate_drifted_data(n_rows: int = 4000, seed: int = 2025):
    df = generate_churn_data(n_rows=n_rows, seed=seed).copy()
    rng = np.random.default_rng(seed)

    # 1) Prices rose ~30%.
    df["monthly_charges"] = (df["monthly_charges"] * 1.30).round(2)
    df["total_charges"] = (df["total_charges"] * 1.30).round(2)

    # 2) Shift ~25% of customers onto Month-to-month contracts.
    flip = rng.random(len(df)) < 0.25
    df.loc[flip, "contract_type"] = "Month-to-month"

    # 3) More support calls.
    df["num_support_calls"] = df["num_support_calls"] + rng.poisson(1.0, size=len(df))

    return df


def main() -> None:
    df = simulate_drifted_data()
    df.to_csv(NEW_DATA_FILE, index=False)
    logger.info("Wrote %d rows of drifted data -> %s", len(df), NEW_DATA_FILE)
    logger.info("Churn rate in new data: %.1f%%", df["churned"].mean() * 100)


if __name__ == "__main__":
    main()
