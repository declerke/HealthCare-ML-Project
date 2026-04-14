"""
routes.py — API endpoint definitions.

Endpoints:
  GET  /           → Serves frontend/index.html
  GET  /health     → Liveness check + model status
  POST /predict    → Run inference on patient data
  POST /retrain    → Trigger model retraining (protected by API key)
"""

import os
import sys
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.model_loader as model_loader
from app.schemas import HealthResponse, PredictRequest, PredictResponse, RetrainResponse
from app.utils import verify_retrain_key
from ml.predict import predict_test_result

router = APIRouter()

FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "index.html"


@router.get("/", include_in_schema=False)
def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(FRONTEND_PATH)


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded(),
        model_version=model_loader.current_version,
    )


@router.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Try again shortly or trigger /retrain.",
        )

    payload = {
        "age": request.age,
        "billing_amount": request.billing_amount,
        "gender": request.gender,
        "blood_type": request.blood_type,
        "medical_condition": request.medical_condition,
        "insurance_provider": request.insurance_provider,
        "admission_type": request.admission_type,
        "medication": request.medication,
    }

    result = predict_test_result(
        payload,
        model_loader.clf,
        model_loader.label_encoders,
        model_loader.scaler,
    )
    result["model_version"] = model_loader.current_version
    return PredictResponse(**result)


@router.post("/retrain", response_model=RetrainResponse, tags=["Training"])
def retrain(x_api_key: str = Header(..., description="RETRAIN_API_KEY value from .env")):
    if not verify_retrain_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )

    from ml.train import train
    metrics = train(log_to_db=True)
    model_loader.reload()

    return RetrainResponse(
        status="success",
        version=metrics["version"],
        accuracy=round(metrics["accuracy"], 4),
        f1_macro=round(metrics["f1_macro"], 4),
        n_samples=metrics["n_samples"],
    )
