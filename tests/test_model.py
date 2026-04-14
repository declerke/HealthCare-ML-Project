"""
test_model.py — Unit tests for preprocessing and inference logic.
Runs without a database or live API; only requires model artifacts.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocess import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    load_encoders,
)
from ml.predict import predict_test_result

VALID_PAYLOAD = {
    "age": 45,
    "billing_amount": 2000.50,
    "gender": "Male",
    "blood_type": "A+",
    "medical_condition": "Diabetes",
    "insurance_provider": "Cigna",
    "admission_type": "Emergency",
    "medication": "Aspirin",
}

EXPECTED_CLASSES = {"Normal", "Abnormal", "Inconclusive"}


@pytest.fixture(scope="module")
def encoders_and_scaler():
    label_encoders, scaler = load_encoders()
    return label_encoders, scaler


@pytest.fixture(scope="module")
def loaded_model():
    import joblib
    model_path = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
    return joblib.load(model_path)


def test_encoders_load(encoders_and_scaler):
    label_encoders, scaler = encoders_and_scaler
    for col in CATEGORICAL_FEATURES + ["test_results"]:
        assert col in label_encoders, f"Missing encoder for column: {col}"


def test_scaler_shape(encoders_and_scaler):
    _, scaler = encoders_and_scaler
    assert scaler.n_features_in_ == len(NUMERIC_FEATURES)


def test_model_loads(loaded_model):
    assert loaded_model is not None


def test_model_classes(loaded_model, encoders_and_scaler):
    label_encoders, _ = encoders_and_scaler
    target_encoder = label_encoders["test_results"]
    assert set(target_encoder.classes_) == EXPECTED_CLASSES


def test_predict_returns_valid_class(loaded_model, encoders_and_scaler):
    label_encoders, scaler = encoders_and_scaler
    result = predict_test_result(VALID_PAYLOAD, loaded_model, label_encoders, scaler)
    assert result["predicted_result"] in EXPECTED_CLASSES


def test_predict_confidence_range(loaded_model, encoders_and_scaler):
    label_encoders, scaler = encoders_and_scaler
    result = predict_test_result(VALID_PAYLOAD, loaded_model, label_encoders, scaler)
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_probabilities_sum_to_one(loaded_model, encoders_and_scaler):
    label_encoders, scaler = encoders_and_scaler
    result = predict_test_result(VALID_PAYLOAD, loaded_model, label_encoders, scaler)
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-4


def test_predict_probabilities_keys(loaded_model, encoders_and_scaler):
    label_encoders, scaler = encoders_and_scaler
    result = predict_test_result(VALID_PAYLOAD, loaded_model, label_encoders, scaler)
    assert set(result["probabilities"].keys()) == EXPECTED_CLASSES


def test_feature_col_count():
    assert len(FEATURE_COLS) == 8
