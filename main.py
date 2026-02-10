from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, conint, confloat

from .model import PatientFeatures, predict_risk


class PatientInput(BaseModel):
    age: conint(ge=18, le=100) = Field(..., description="Age in years")
    sex: conint(ge=0, le=1) = Field(..., description="0 = female, 1 = male")
    bmi: confloat(ge=10, le=60) = Field(..., description="Body Mass Index")
    systolic_bp: confloat(ge=80, le=250) = Field(..., description="Systolic blood pressure in mmHg")
    diastolic_bp: confloat(ge=40, le=150) = Field(..., description="Diastolic blood pressure in mmHg")
    cholesterol: confloat(ge=80, le=400) = Field(..., description="Total cholesterol level (mg/dL)")
    smoker: conint(ge=0, le=1) = Field(..., description="0 = non-smoker, 1 = smoker")
    family_history: conint(ge=0, le=1) = Field(..., description="0 = no, 1 = yes")
    exercise_level: conint(ge=0, le=2) = Field(
        ..., description="0 = low, 1 = moderate, 2 = high"
    )


class PredictionResponse(BaseModel):
    risk_probability: float
    risk_category: str
    recommendation: str


app = FastAPI(
    title="HealthTech Early Risk API",
    description="API to predict early cardiovascular-like health risk using basic patient data.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    # When serving the frontend from this backend, the origin will match the API.
    # During local development we still allow all origins for simplicity.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the static frontend so that backend and frontend are naturally linked
app.mount(
    "/app",
    StaticFiles(directory="frontend", html=True),
    name="frontend",
)


@app.get("/", include_in_schema=False)
def serve_index():
    """
    Convenience route so you can open http://localhost:8000
    and immediately see the web UI.
    """
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    features = PatientFeatures(
        age=patient.age,
        sex=patient.sex,
        bmi=patient.bmi,
        systolic_bp=patient.systolic_bp,
        diastolic_bp=patient.diastolic_bp,
        cholesterol=patient.cholesterol,
        smoker=patient.smoker,
        family_history=patient.family_history,
        exercise_level=patient.exercise_level,
    )

    result = predict_risk(features)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

