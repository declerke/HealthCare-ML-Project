import os
from pathlib import Path
from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

_BASE = Path(__file__).resolve().parents[1]
MODEL_PATH    = Path(os.getenv("MODEL_PATH",    str(_BASE / "models" / "model.joblib")))
ENCODERS_PATH = Path(os.getenv("ENCODERS_PATH", str(_BASE / "models" / "encoders.joblib")))

clf:            Optional[RandomForestClassifier] = None
label_encoders: Optional[dict]                   = None
scaler:          Optional[StandardScaler]          = None
current_version: Optional[str]                   = None


def load() -> None:
    global clf, label_encoders, scaler, current_version

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python ml/train.py` to generate it."
        )
    if not ENCODERS_PATH.exists():
        raise FileNotFoundError(
            f"Encoders not found at {ENCODERS_PATH}. "
            "Run `python ml/train.py` to generate them."
        )

    clf = joblib.load(MODEL_PATH)
    encoder_data = joblib.load(ENCODERS_PATH)
    label_encoders = encoder_data["label_encoders"]
    scaler         = encoder_data["scaler"]
    current_version = MODEL_PATH.stat().st_mtime.__str__()[:10]
    print(f"Model loaded from {MODEL_PATH}")


def reload() -> None:
    load()


def is_loaded() -> bool:
    return clf is not None
