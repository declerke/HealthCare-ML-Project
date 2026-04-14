import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml.preprocess import encode_single_row


def predict_test_result(
    payload: dict,
    clf: RandomForestClassifier,
    label_encoders: dict,
    scaler: StandardScaler,
) -> dict:
    X = encode_single_row(payload, label_encoders, scaler)

    predicted_class_idx = clf.predict(X)[0]
    probabilities = clf.predict_proba(X)[0]

    target_encoder: LabelEncoder = label_encoders["test_results"]
    predicted_label = target_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = float(probabilities[predicted_class_idx])

    class_probs = {
        label: float(prob)
        for label, prob in zip(target_encoder.classes_, probabilities)
    }

    return {
        "predicted_result": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in class_probs.items()},
    }
