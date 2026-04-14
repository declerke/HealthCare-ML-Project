import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_healthcare.csv"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/healthcare_db")


def load_patients() -> None:
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded {len(df):,} rows from {CLEAN_PATH}")

    df = df.rename(columns={
        "Name": "name",
        "Age": "age",
        "Gender": "gender",
        "Blood Type": "blood_type",
        "Medical Condition": "medical_condition",
        "Date of Admission": "date_of_admission",
        "Doctor": "doctor",
        "Hospital": "hospital",
        "Insurance Provider": "insurance_provider",
        "Billing Amount": "billing_amount",
        "Room Number": "room_number",
        "Admission Type": "admission_type",
        "Discharge Date": "discharge_date",
        "Medication": "medication",
        "Test Results": "test_results",
    })

    engine = create_engine(DATABASE_URL)

    with engine.begin() as conn:
        existing = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
        print(f"Existing rows in patients table: {existing:,}")

    df.to_sql("patients_staging", engine, if_exists="replace", index=False)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO patients (
                name, age, gender, blood_type, medical_condition,
                date_of_admission, doctor, hospital, insurance_provider,
                billing_amount, room_number, admission_type,
                discharge_date, medication, test_results
            )
            SELECT
                name, age, gender, blood_type, medical_condition,
                date_of_admission::date, doctor, hospital, insurance_provider,
                billing_amount, room_number, admission_type,
                discharge_date::date, medication, test_results
            FROM patients_staging
            ON CONFLICT DO NOTHING
        """))
        conn.execute(text("DROP TABLE IF EXISTS patients_staging"))

    with engine.begin() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
        print(f"Total rows in patients table after load: {total:,}")


if __name__ == "__main__":
    load_patients()
