"""
test_api.py — Integration tests for all FastAPI endpoints.
Uses httpx TestClient with lifespan; no external services required.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app

VALID_PAYLOAD = {
    "age": 45,
    "gender": "Male",
    "blood_type": "A+",
    "admission_type": "Emergency",
    "billing_amount": 2000.50,
    "insurance_provider": "Cigna",
    "medical_condition": "Diabetes",
    "medication": "Aspirin",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_returns_200(client):
    res = client.get("/health")
    assert res.status_code == 200


def test_health_model_loaded(client):
    res = client.get("/health")
    body = res.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_200(client):
    res = client.post("/predict", json=VALID_PAYLOAD)
    assert res.status_code == 200


def test_predict_response_structure(client):
    res = client.post("/predict", json=VALID_PAYLOAD)
    body = res.json()
    assert "predicted_result" in body
    assert "confidence" in body
    assert "probabilities" in body


def test_predict_result_is_valid_class(client):
    res = client.post("/predict", json=VALID_PAYLOAD)
    result = res.json()["predicted_result"]
    assert result in {"Normal", "Abnormal", "Inconclusive"}


def test_predict_confidence_range(client):
    res = client.post("/predict", json=VALID_PAYLOAD)
    confidence = res.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_probabilities_sum_to_one(client):
    res = client.post("/predict", json=VALID_PAYLOAD)
    probs = res.json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-4


def test_predict_missing_field_returns_422(client):
    incomplete = {k: v for k, v in VALID_PAYLOAD.items() if k != "age"}
    res = client.post("/predict", json=incomplete)
    assert res.status_code == 422


def test_predict_invalid_gender_returns_422(client):
    bad = {**VALID_PAYLOAD, "gender": "Unknown"}
    res = client.post("/predict", json=bad)
    assert res.status_code == 422


def test_predict_invalid_blood_type_returns_422(client):
    bad = {**VALID_PAYLOAD, "blood_type": "C+"}
    res = client.post("/predict", json=bad)
    assert res.status_code == 422


def test_predict_negative_age_returns_422(client):
    bad = {**VALID_PAYLOAD, "age": -5}
    res = client.post("/predict", json=bad)
    assert res.status_code == 422


def test_retrain_without_key_returns_422(client):
    res = client.post("/retrain")
    assert res.status_code == 422


def test_retrain_with_wrong_key_returns_401(client):
    res = client.post("/retrain", headers={"x-api-key": "wrong_key"})
    assert res.status_code == 401


def test_frontend_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Healthcare Test Result Predictor" in res.text
