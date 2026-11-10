"""Pydantic models: they define and validate the API input/output schema."""
from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    """Input of POST /predict. The 10 features of the scikit-learn Diabetes
    dataset (already mean-centered and scaled, as in the original data)."""

    model_config = ConfigDict(
        extra="forbid",  # unknown fields -> 422
        json_schema_extra={
            "example": {
                "age": 0.0381, "sex": 0.0507, "bmi": 0.0617, "bp": 0.0219,
                "s1": -0.0442, "s2": -0.0348, "s3": -0.0434, "s4": -0.0026,
                "s5": 0.0199, "s6": -0.0176,
            }
        },
    )

    age: float = Field(..., description="Age (normalized)")
    sex: float = Field(..., description="Sex (normalized)")
    bmi: float = Field(..., description="Body mass index (normalized)")
    bp: float = Field(..., description="Average blood pressure (normalized)")
    s1: float = Field(..., description="Total serum cholesterol (normalized)")
    s2: float = Field(..., description="LDL cholesterol (normalized)")
    s3: float = Field(..., description="HDL cholesterol (normalized)")
    s4: float = Field(..., description="Total cholesterol / HDL (normalized)")
    s5: float = Field(..., description="Log of serum triglycerides (normalized)")
    s6: float = Field(..., description="Blood sugar level (normalized)")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow "model_*" field names

    prediction: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow "model_*" field names

    model_name: str
    model_version: str
    alias: str
    run_id: str | None = None
    model_type: str | None = None
    metrics: dict[str, float] = {}
    rmse_threshold: float | None = None
    features: list[str]
