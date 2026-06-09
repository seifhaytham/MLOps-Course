"""Request and response schemas for the churn prediction API."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single customer record fed to the classifier.

    Field names mirror the columns the trained ColumnTransformer expects;
    see [src/preprocessing.py](src/preprocessing.py) for the canonical list.
    """

    CreditScore: int = Field(..., ge=0, le=1000, examples=[619])
    Geography: Literal["France", "Spain", "Germany"] = Field(..., examples=["France"])
    Gender: Literal["Male", "Female"] = Field(..., examples=["Female"])
    Age: int = Field(..., ge=0, le=120, examples=[42])
    Tenure: int = Field(..., ge=0, le=20, examples=[2])
    Balance: float = Field(..., ge=0.0, examples=[0.0])
    NumOfProducts: int = Field(..., ge=1, le=4, examples=[1])
    HasCrCard: int = Field(..., ge=0, le=1, examples=[1])
    IsActiveMember: int = Field(..., ge=0, le=1, examples=[1])
    EstimatedSalary: float = Field(..., ge=0.0, examples=[101348.88])


class PredictResponse(BaseModel):
    """Classifier output."""

    prediction: int = Field(..., description="1 if churn, 0 otherwise")
    probability: float = Field(..., ge=0.0, le=1.0, description="P(Exited=1)")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool


class HomeResponse(BaseModel):
    service: str
    version: str
    docs: str
