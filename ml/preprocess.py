"""
preprocess.py — Feature engineering and encoding for training and inference.

ML features used (8 total):
  Numeric  : age, billing_amount
  Categorical: gender, blood_type, medical_condition,
               insurance_provider, admission_type, medication

Target: test_results  (Normal / Abnormal / Inconclusive)

Encoders are saved alongside the model so inference always uses
the same vocabulary as training.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_COLS = [
    "age",
    "billing_amount",
    "gender",
    "blood_type",
    "medical_condition",
    "insurance_provider",
    "admission_type",
    "medication",
]

CATEGORICAL_FEATURES = [
    "gender",
    "blood_type",
    "medical_condition",
    "insurance_provider",
    "admission_type",
    "medication",
]

NUMERIC_FEATURES = ["age", "billing_amount"]
TARGET_COL = "test_results"

ENCODERS_PATH = Path(__file__).resolve().parents[1] / "models" / "encoders.joblib"


def build_encoders(df: pd.DataFrame) -> Tuple[dict, StandardScaler]:
    label_encoders: dict = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col])
        label_encoders[col] = le

    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_FEATURES].values)

    target_encoder = LabelEncoder()
    target_encoder.fit(df[TARGET_COL])
    label_encoders[TARGET_COL] = target_encoder

    return label_encoders, scaler


def save_encoders(label_encoders: dict, scaler: StandardScaler) -> None:
    ENCODERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"label_encoders": label_encoders, "scaler": scaler}, ENCODERS_PATH)
    print(f"Encoders saved to {ENCODERS_PATH}")


def load_encoders() -> Tuple[dict, StandardScaler]:
    data = joblib.load(ENCODERS_PATH)
    return data["label_encoders"], data["scaler"]


def encode_features(
    df: pd.DataFrame,
    label_encoders: dict,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        df[col] = label_encoders[col].transform(df[col])

    numeric_scaled = scaler.transform(df[NUMERIC_FEATURES].values)
    for i, col in enumerate(NUMERIC_FEATURES):
        df[col] = numeric_scaled[:, i]

    X = df[FEATURE_COLS].values
    y = label_encoders[TARGET_COL].transform(df[TARGET_COL].values)
    return X, y


def encode_single_row(row: dict, label_encoders: dict, scaler: StandardScaler) -> np.ndarray:
    """Encode a single inference payload dict → 1-row numpy array."""
    df = pd.DataFrame([row])
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    for col in CATEGORICAL_FEATURES:
        df[col] = label_encoders[col].transform(df[col])

    numeric_scaled = scaler.transform(df[NUMERIC_FEATURES].values)
    for i, col in enumerate(NUMERIC_FEATURES):
        df[col] = numeric_scaled[:, i]

    return df[FEATURE_COLS].values
