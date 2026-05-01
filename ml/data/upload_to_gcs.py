"""Upload prepared dataset to GCS.

Run:
    GCS_ML_DATA_BUCKET=your-bucket python ml/data/upload_to_gcs.py
"""
import os
from pathlib import Path
from google.cloud import storage

DATASET_DIR = Path("ml/data/processed/yolo_dataset")


def upload(bucket_name: str, prefix: str = "datasets/yolo_person") -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files = [f for f in DATASET_DIR.rglob("*") if f.is_file()]
    for f in files:
        blob_name = f"{prefix}/{f.relative_to(DATASET_DIR)}"
        bucket.blob(blob_name).upload_from_filename(str(f))
        print(f"  Uploaded gs://{bucket_name}/{blob_name}")
    print(f"Done. {len(files)} files uploaded.")


if __name__ == "__main__":
    bucket = os.environ["GCS_ML_DATA_BUCKET"]
    upload(bucket)
