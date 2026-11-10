"""API tests with a fake model (no MLflow registry needed).

TestClient is used WITHOUT a ``with`` block on purpose: that way the startup
hook (which would try to load the real champion model) does not run, and we
control the service state from each test.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import service

VALID_PAYLOAD = {
    "age": 0.0381, "sex": 0.0507, "bmi": 0.0617, "bp": 0.0219, "s1": -0.0442,
    "s2": -0.0348, "s3": -0.0434, "s4": -0.0026, "s5": 0.0199, "s6": -0.0176,
}


class FakeModel:
    def predict(self, df):
        return np.array([123.45] * len(df))


class BrokenModel:
    def predict(self, df):
        raise RuntimeError("boom")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_service():
    """Put a fake champion model in the shared service, restore afterwards."""
    saved = dict(service.__dict__)
    service.model = FakeModel()
    service.version = "7"
    service.run_id = "abc123"
    service.model_type = "FakeModel"
    service.metrics = {"rmse": 50.0, "mae": 40.0, "r2": 0.5}
    yield service
    service.__dict__.update(saved)


@pytest.fixture
def unloaded_service():
    saved = dict(service.__dict__)
    service.model = None
    yield service
    service.__dict__.update(saved)


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ok_even_without_model(client, unloaded_service):
    assert client.get("/health").status_code == 200


def test_model_info(client, loaded_service):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "diabetes-progression-regressor"
    assert body["model_version"] == "7"
    assert body["alias"] == "champion"
    assert body["metrics"]["rmse"] == 50.0


def test_model_info_503_without_model(client, unloaded_service):
    assert client.get("/model-info").status_code == 503


def test_predict_valid_input(client, loaded_service):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == pytest.approx(123.45)
    assert body["model_version"] == "7"


def test_predict_missing_field_returns_422(client, loaded_service):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "bmi"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_wrong_type_returns_422(client, loaded_service):
    payload = {**VALID_PAYLOAD, "age": "not-a-number"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_unknown_field_returns_422(client, loaded_service):
    payload = {**VALID_PAYLOAD, "unexpected": 1.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_503_when_model_not_loaded(client, unloaded_service):
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 503


def test_predict_500_when_prediction_fails(client, loaded_service):
    loaded_service.model = BrokenModel()
    assert client.post("/predict", json=VALID_PAYLOAD).status_code == 500
