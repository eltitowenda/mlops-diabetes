"""FastAPI application serving the champion model.

Run (from the repository root):
    uvicorn src.api.main:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.api.service import service
from src.config import FEATURE_COLUMNS, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to load the champion model at startup. If it fails the API still
    # starts (so /health works) and /predict answers 503.
    service.load()
    yield


app = FastAPI(
    title="Diabetes Progression API",
    description="Serves the champion model of the MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


def require_model():
    """Return the loaded model or raise 503 Service Unavailable."""
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available: no champion model is loaded. "
                   "Run the training pipeline and restart the API.",
        )
    return service.model


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check: 200 OK while the service is running."""
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Which model is being served (name, version, alias, metrics...)."""
    require_model()
    return service.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    """Predict disease progression for one patient.

    - 200: prediction returned
    - 422: invalid input (missing field / wrong type / unknown field)
    - 503: champion model not loaded
    - 500: unexpected failure while predicting
    """
    model = require_model()
    try:
        data = pd.DataFrame([patient.model_dump()], columns=FEATURE_COLUMNS)
        prediction = float(np.asarray(model.predict(data)).ravel()[0])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error while generating the prediction.")

    return {
        "prediction": prediction,
        "model_name": service.name,
        "model_version": service.version,
    }
