"""
models.py — SQLAlchemy ORM table definitions matching queries.sql.
"""

from datetime import date, datetime
from sqlalchemy import Boolean, Column, Date, Float, Integer, Numeric, String, DateTime
from database.db_connection import Base


class Patient(Base):
    __tablename__ = "patients"

    id                 = Column(Integer, primary_key=True, index=True)
    name               = Column(String(120))
    age                = Column(Integer, nullable=False)
    gender             = Column(String(10), nullable=False)
    blood_type         = Column(String(5), nullable=False)
    medical_condition  = Column(String(60), nullable=False)
    date_of_admission  = Column(Date)
    doctor             = Column(String(120))
    hospital           = Column(String(120))
    insurance_provider = Column(String(60))
    billing_amount     = Column(Numeric(12, 2))
    room_number        = Column(Integer)
    admission_type     = Column(String(20), nullable=False)
    discharge_date     = Column(Date)
    medication         = Column(String(60))
    test_results       = Column(String(20), nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id                 = Column(Integer, primary_key=True, index=True)
    age                = Column(Integer, nullable=False)
    gender             = Column(String(10), nullable=False)
    blood_type         = Column(String(5), nullable=False)
    medical_condition  = Column(String(60), nullable=False)
    admission_type     = Column(String(20), nullable=False)
    billing_amount     = Column(Numeric(12, 2), nullable=False)
    insurance_provider = Column(String(60), nullable=False)
    medication         = Column(String(60), nullable=False)
    predicted_result   = Column(String(20), nullable=False)
    confidence         = Column(Numeric(5, 4), nullable=False)
    model_version      = Column(String(50))
    created_at         = Column(DateTime, default=datetime.utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id         = Column(Integer, primary_key=True, index=True)
    version    = Column(String(50), nullable=False)
    accuracy   = Column(Numeric(6, 4), nullable=False)
    f1_score   = Column(Numeric(6, 4), nullable=False)
    n_samples  = Column(Integer, nullable=False)
    is_active  = Column(Boolean, default=True)
    trained_at = Column(DateTime, default=datetime.utcnow)
