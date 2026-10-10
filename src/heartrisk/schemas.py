"""Request / response contracts for the prediction API.

Ranges follow the UCI documentation, widened to plausible clinical limits so that
unusual-but-real patients are accepted while typos (e.g. age=630) are rejected.
`ca` and `thal` may be omitted: the model pipeline imputes them, exactly as it
did for the 6 missing values in the training data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT})

    age: int = Field(..., ge=18, le=110, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=60, le=260, description="Resting blood pressure, mm Hg")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol, mg/dl")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate achieved, bpm")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise, mm")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST: 1 up, 2 flat, 3 down")
    ca: Literal[0, 1, 2, 3] | None = Field(None, description="Major vessels coloured by fluoroscopy (optional)")
    thal: Literal[3, 6, 7] | None = Field(
        None, description="Thalassemia: 3 normal, 6 fixed defect, 7 reversible defect (optional)")


class BatchRequest(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(..., description="1 = heart disease predicted")
    label: Literal["disease", "no disease"]
    probability_disease: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0.5, le=1, description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]


class PredictResponse(Prediction):
    model_name: str
    model_version: str
    request_id: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_name: str
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    uptime_seconds: float
