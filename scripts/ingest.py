"""
ingest.py — Download the Kaggle healthcare dataset and copy the raw CSV
into data/ so subsequent scripts can find it at a stable path.
"""

import os
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATASET_SLUG = "prasad22/healthcare-dataset"
RAW_FILE = RAW_DIR / "healthcare_dataset.csv"
DEST_FILE = DATA_DIR / "healthcare.csv"


def download_dataset() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_FILE.exists():
        print(f"Raw file already exists: {RAW_FILE}")
    else:
        print(f"Downloading dataset '{DATASET_SLUG}' from Kaggle ...")
        os.system(f'kaggle datasets download -d {DATASET_SLUG} -p "{RAW_DIR}" --unzip')
        if not RAW_FILE.exists():
            raise FileNotFoundError(
                f"Download completed but expected file not found: {RAW_FILE}\n"
                "Check that ~/.kaggle/kaggle.json is present and valid."
            )
        print(f"Downloaded: {RAW_FILE}")

    shutil.copy(RAW_FILE, DEST_FILE)
    print(f"Copied to: {DEST_FILE}")


if __name__ == "__main__":
    download_dataset()
