import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocess import (
    FEATURE_COLS,
    TARGET_COL,
    build_encoders,
    encode_features,
    save_encoders,
)
from ml.evaluate import evaluate_model

load_dotenv()

CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_healthcare.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/healthcare_db")


def _build_candidates(n_classes: int) -> list[tuple[str, object]]:
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    return [("XGBoost", xgb), ("RandomForest", rf)]


def train(log_to_db: bool = False) -> dict:
    print(f"[{datetime.utcnow().isoformat()}] Starting training run ...")

    df = pd.read_csv(CLEAN_PATH)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded {len(df):,} rows")

    label_encoders, scaler = build_encoders(df)
    X, y = encode_features(df, label_encoders, scaler)

    n_classes = len(label_encoders[TARGET_COL].classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    candidates = _build_candidates(n_classes)
    best_clf = None
    best_metrics: dict = {}
    all_model_metrics: dict = {}

    for name, clf in candidates:
        print(f"\nTraining {name} ...")
        clf.fit(X_train, y_train)
        metrics = evaluate_model(clf, X_test, y_test, label_encoders[TARGET_COL])
        all_model_metrics[name] = metrics
        print(
            f"  {name}  Accuracy: {metrics['accuracy']:.4f}  |  "
            f"F1 (macro): {metrics['f1_macro']:.4f}"
        )
        if best_clf is None or metrics["f1_macro"] > best_metrics.get("f1_macro", 0):
            best_clf = clf
            best_metrics = metrics
            best_metrics["model_name"] = name

    print(f"\nBest model: {best_metrics['model_name']}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, MODEL_PATH, compress=3)
    save_encoders(label_encoders, scaler)
    print(f"Model saved to {MODEL_PATH}")

    version = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    best_metrics["version"] = version
    best_metrics["n_samples"] = len(df)
    best_metrics["all_models"] = {
        name: {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]}
        for name, m in all_model_metrics.items()
    }

    if log_to_db:
        _log_version_to_db(best_metrics)

    print(f"Training complete — version {version}")
    return best_metrics


def _log_version_to_db(metrics: dict) -> None:
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(DATABASE_URL)
        with engine.begin() as conn:
            conn.execute(text("UPDATE model_versions SET is_active = FALSE"))
            conn.execute(
                text("""
                    INSERT INTO model_versions (version, accuracy, f1_score, n_samples, is_active)
                    VALUES (:version, :accuracy, :f1_score, :n_samples, TRUE)
                """),
                {
                    "version": metrics["version"],
                    "accuracy": float(metrics["accuracy"]),
                    "f1_score": float(metrics["f1_macro"]),
                    "n_samples": metrics["n_samples"],
                },
            )
        print(f"Model version {metrics['version']} logged to database")
    except Exception as exc:
        print(f"Warning: could not log to DB — {exc}")


if __name__ == "__main__":
    train(log_to_db=False)
