"""
Pydantic request/response models = the API's public contract.

WHY THIS MATTERS
----------------
These classes are the firewall between the messy outside world and your model.
- `Literal[...]` on categorical fields means an invalid category (e.g. a typo
  like "Fibre optic") is rejected with a clear 422 error BEFORE it reaches the
  model. No silent garbage-in-garbage-out.
- Numeric constraints (ge=0, gt=0) catch impossible values (negative tenure).
- `json_schema_extra` examples power the auto-generated Swagger docs at /docs,
  so consumers see a valid request without reading your code.

The field names MUST match src/config.FEATURE_COLUMNS exactly -- that's the whole
point of a shared schema.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --- Allowed categorical values (must match the training data) ---
Gender = Literal["Male", "Female"]
YesNo = Literal["Yes", "No"]
Contract = Literal["Month-to-month", "One year", "Two year"]
Payment = Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
Internet = Literal["DSL", "Fiber optic", "No"]


class CustomerFeatures(BaseModel):
    """One customer's features. All fields required and validated."""

    senior_citizen: Literal[0, 1] = Field(..., description="1 if senior citizen else 0")
    tenure_months: int = Field(..., ge=0, le=100, description="Months as a customer")
    monthly_charges: float = Field(..., gt=0, description="Monthly charge amount")
    total_charges: float = Field(..., ge=0, description="Lifetime charges")
    num_support_calls: int = Field(..., ge=0, description="Support calls made")

    gender: Gender
    partner: YesNo
    dependents: YesNo
    contract_type: Contract
    payment_method: Payment
    internet_service: Internet
    tech_support: YesNo
    paperless_billing: YesNo

    model_config = {
        "json_schema_extra": {
            "example": {
                "senior_citizen": 0,
                "tenure_months": 2,
                "monthly_charges": 95.5,
                "total_charges": 190.0,
                "num_support_calls": 4,
                "gender": "Female",
                "partner": "No",
                "dependents": "No",
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
                "tech_support": "No",
                "paperless_billing": "Yes",
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = churn, 0 = stay")
    prediction_label: Literal["churn", "stay"]
    churn_probability: float = Field(..., ge=0, le=1)
    model_name: str
    model_version: str


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_source: str
    metrics: dict
    features: List[str]
    trained_at: Optional[str] = None
