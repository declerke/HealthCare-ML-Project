"""
evaluate.py — Model evaluation utilities.
Returns a metrics dict; also prints the full classification report.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray, target_encoder: LabelEncoder) -> dict:
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    class_names = target_encoder.classes_

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    header = f"{'':15s}" + "".join(f"{c:>15s}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:15s}" + "".join(f"{v:>15d}" for v in row))
    print()

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
    }
