from pathlib import Path
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "healthcare.csv"
CLEAN_PATH = Path(__file__).resolve().parents[1] / "data" / "cleaned_healthcare.csv"

CATEGORICAL_COLS = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
]

DATE_COLS = ["Date of Admission", "Discharge Date"]
STRING_COLS = ["Name", "Doctor", "Hospital"]


def clean_dataset() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df):,} rows from {RAW_PATH}")

    for col in STRING_COLS:
        df[col] = df[col].str.strip().str.title()

    for col in CATEGORICAL_COLS:
        df[col] = df[col].str.strip().str.title()

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d")

    df["Billing Amount"] = df["Billing Amount"].round(2)

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped:,} duplicate rows")

    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Nulls detected after cleaning:\n{null_counts[null_counts > 0]}")

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {len(df):,} clean rows to {CLEAN_PATH}")
    print(f"\nTest Results distribution:\n{df['Test Results'].value_counts()}")
    return df


if __name__ == "__main__":
    clean_dataset()
