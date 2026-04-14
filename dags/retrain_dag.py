"""
Healthcare ML — Weekly Retraining DAG  (Apache Airflow 3)
=========================================================
Mirrors the GitHub Actions cron (0 12 * * 6) so the complete retraining
pipeline can be orchestrated, monitored, and triggered manually via the
Airflow UI when running locally with Docker Compose.

Pipeline tasks
--------------
1. load_data          — validate the cleaned dataset exists and report row count
2. train_model        — fit RandomForestClassifier, save joblib artifacts, log version to DB
3. validate_artifacts — confirm model.joblib was written and can be deserialised

XCom values flow automatically between tasks via TaskFlow API function returns.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from airflow.decorators import dag, task

# ---------------------------------------------------------------------------
# Project root resolution
#   • Inside Docker (docker-compose):  /opt/airflow/project  (via env var)
#   • Local / pytest:  repo root derived from this file's location
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.getenv("AIRFLOW_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
)
sys.path.insert(0, str(PROJECT_ROOT))


@dag(
    dag_id="healthcare_retrain",
    schedule="0 12 * * 6",          # Every Saturday at 12:00 UTC — matches GitHub Actions
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "healthcare", "retrain"],
    doc_md=__doc__,
)
def healthcare_retrain() -> None:
    """Healthcare ML weekly retraining pipeline."""

    # ------------------------------------------------------------------
    # Task 1: validate the cleaned dataset is present and readable
    # ------------------------------------------------------------------
    @task(task_id="load_data")
    def load_data() -> dict:
        """Load cleaned_healthcare.csv and return row-count metadata via XCom."""
        import pandas as pd

        clean_path = PROJECT_ROOT / "data" / "cleaned_healthcare.csv"
        if not clean_path.exists():
            raise FileNotFoundError(
                f"Cleaned dataset not found at {clean_path}. "
                "Run scripts/ingest.py and scripts/clean.py first."
            )

        df = pd.read_csv(clean_path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        n_rows = len(df)

        print(f"[load_data] Dataset OK — {n_rows:,} rows loaded from {clean_path}")
        return {"n_samples": n_rows, "clean_path": str(clean_path)}

    # ------------------------------------------------------------------
    # Task 2: train the model and persist artifacts
    # ------------------------------------------------------------------
    @task(task_id="train_model")
    def train_model(data_info: dict) -> dict:
        """Fit RandomForestClassifier, save model.joblib + encoders.joblib, log to DB."""
        from ml.train import train  # noqa: PLC0415  (import inside task for Airflow isolation)

        print(f"[train_model] Starting training on {data_info['n_samples']:,} samples ...")
        metrics = train(log_to_db=True)

        print(
            f"[train_model] Done\n"
            f"  Version  : {metrics['version']}\n"
            f"  Accuracy : {metrics['accuracy']:.4f}\n"
            f"  F1 macro : {metrics['f1_macro']:.4f}\n"
            f"  Samples  : {metrics['n_samples']:,}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Task 3: confirm artifacts were written and can be deserialised
    # ------------------------------------------------------------------
    @task(task_id="validate_artifacts")
    def validate_artifacts(metrics: dict) -> None:
        """Assert model.joblib and encoders.joblib exist and load cleanly."""
        import joblib

        model_path    = PROJECT_ROOT / "models" / "model.joblib"
        encoders_path = PROJECT_ROOT / "models" / "encoders.joblib"

        assert model_path.exists(),    f"model.joblib not found at {model_path}"
        assert encoders_path.exists(), f"encoders.joblib not found at {encoders_path}"

        clf = joblib.load(model_path)
        assert hasattr(clf, "predict_proba"), (
            f"Loaded object is not a sklearn classifier: {type(clf)}"
        )

        size_mb = model_path.stat().st_size / 1_048_576
        print(
            f"[validate_artifacts] Artifacts OK\n"
            f"  Version    : {metrics['version']}\n"
            f"  Model size : {size_mb:.1f} MB\n"
            f"  Classes    : {clf.classes_.tolist()}"
        )

    # ------------------------------------------------------------------
    # DAG wiring — TaskFlow XCom propagates return values automatically
    # ------------------------------------------------------------------
    data_info = load_data()
    metrics   = train_model(data_info)
    validate_artifacts(metrics)


# Instantiate the DAG (required at module level for Airflow to register it)
healthcare_retrain()
