from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


GENDERS          = Literal["Male", "Female"]
BLOOD_TYPES      = Literal["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
ADMISSION_TYPES  = Literal["Emergency", "Elective", "Urgent"]
MEDICAL_CONDITIONS = Literal["Diabetes", "Hypertension", "Asthma", "Obesity", "Arthritis", "Cancer"]
INSURANCE_PROVIDERS = Literal["Medicare", "Aetna", "UnitedHealthcare", "Cigna", "Blue Cross"]
MEDICATIONS = Literal["Aspirin", "Ibuprofen", "Paracetamol", "Penicillin", "Lipitor"]
TEST_RESULTS = Literal["Normal", "Abnormal", "Inconclusive"]


class PredictRequest(BaseModel):
    age: int = Field(..., ge=1, le=120)
    gender: GENDERS
    blood_type: BLOOD_TYPES
    admission_type: ADMISSION_TYPES
    billing_amount: float = Field(..., ge=0)
    insurance_provider: INSURANCE_PROVIDERS
    medical_condition: MEDICAL_CONDITIONS
    medication: MEDICATIONS

    model_config = {"json_schema_extra": {"example": {
        "age": 45,
        "gender": "Male",
        "blood_type": "A+",
        "admission_type": "Emergency",
        "billing_amount": 2000.50,
        "insurance_provider": "Cigna",
        "medical_condition": "Diabetes",
        "medication": "Aspirin",
    }}}


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_result: TEST_RESULTS
    confidence: float
    probabilities: Dict[str, float]
    model_version: Optional[str] = None


class RetrainResponse(BaseModel):
    status: str
    version: str
    accuracy: float
    f1_macro: float
    n_samples: int


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
