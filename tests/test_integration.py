"""End-to-end test against the REAL champion model.

Only runs when RUN_INTEGRATION=1 (the CI sets it AFTER running the pipeline,
because it needs a trained and registered champion model).
"""
import os

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 after running the pipeline",
)

PAYLOAD = {
    "age": 0.0381, "sex": 0.0507, "bmi": 0.0617, "bp": 0.0219, "s1": -0.0442,
    "s2": -0.0348, "s3": -0.0434, "s4": -0.0026, "s5": 0.0199, "s6": -0.0176,
}


def test_real_champion_is_served():
    with TestClient(app) as client:  # `with` -> runs startup -> loads champion
        assert client.get("/health").json() == {"status": "ok"}

        info = client.get("/model-info")
        assert info.status_code == 200
        assert info.json()["alias"] == "champion"

        prediction = client.post("/predict", json=PAYLOAD)
        assert prediction.status_code == 200
        assert isinstance(prediction.json()["prediction"], float)

        assert client.post("/predict", json={"age": 0.1}).status_code == 422
