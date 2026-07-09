"""
Generate a realistic synthetic customer-churn dataset (Telco-style).

WHY SYNTHETIC?
--------------
Real churn datasets (e.g. the IBM Telco dataset) are great, but they require a
download and licensing caveats. For a reproducible teaching project we generate
data with *real signal* baked in: contract type, tenure, monthly charges and
support calls genuinely drive churn probability. This means the model actually
has something to learn — unlike pure random noise — and every teammate gets the
exact same data because we fix the random seed.

In a later phase you can swap this for a real CSV; the rest of the pipeline
won't care, because it only depends on the column names defined in config.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE, RAW_DATA_FILE, ALL_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_churn_data(n_rows: int = 7000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a DataFrame of `n_rows` customers with a churn label."""
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Male", "Female"], size=n_rows)
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_rows, p=[0.30, 0.70])

    tenure_months = rng.integers(0, 73, size=n_rows)  # 0..72 months

    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_rows,
        p=[0.55, 0.24, 0.21],
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], size=n_rows, p=[0.34, 0.44, 0.22]
    )
    tech_support = rng.choice(["Yes", "No"], size=n_rows, p=[0.37, 0.63])
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.59, 0.41])

    # Monthly charges depend a bit on internet service (fiber costs more).
    base_charge = np.where(
        internet_service == "Fiber optic",
        rng.normal(85, 15, n_rows),
        np.where(internet_service == "DSL", rng.normal(55, 12, n_rows), rng.normal(25, 8, n_rows)),
    )
    monthly_charges = np.clip(base_charge, 15, 130).round(2)

    # Total charges roughly = monthly * tenure, with noise; new customers => low.
    total_charges = np.clip(
        monthly_charges * np.maximum(tenure_months, 1) * rng.normal(1.0, 0.05, n_rows),
        0,
        None,
    ).round(2)

    num_support_calls = rng.poisson(1.5, size=n_rows)

    # --- The "true" churn logic (this is the signal the model learns) ---
    logit = (
        -1.2
        + 1.4 * (contract_type == "Month-to-month")
        - 0.9 * (contract_type == "Two year")
        + 0.9 * (internet_service == "Fiber optic")
        + 0.6 * (payment_method == "Electronic check")
        - 0.025 * tenure_months
        + 0.012 * (monthly_charges - 65)
        + 0.30 * num_support_calls
        + 0.35 * senior_citizen
        - 0.4 * (tech_support == "Yes")
        + 0.25 * (paperless_billing == "Yes")
    )
    churn_prob = _sigmoid(logit)
    churned = (rng.random(n_rows) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "internet_service": internet_service,
            "tech_support": tech_support,
            "paperless_billing": paperless_billing,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_calls": num_support_calls,
            "churned": churned,
        }
    )
    # Reorder to the canonical column order from config.
    return df[ALL_COLUMNS]


def main() -> None:
    df = generate_churn_data()
    df.to_csv(RAW_DATA_FILE, index=False)
    churn_rate = df["churned"].mean()
    logger.info("Wrote %d rows to %s", len(df), RAW_DATA_FILE)
    logger.info("Churn rate: %.1f%% (a realistic, imbalanced target)", churn_rate * 100)


if __name__ == "__main__":
    main()
