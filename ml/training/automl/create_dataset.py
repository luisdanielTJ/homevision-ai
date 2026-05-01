"""Create a Vertex AI Image Dataset and import images for AutoML training.

The import file (automl_import.jsonl) must exist in GCS. Each line:
    {"imageGcsUri": "gs://bucket/image.jpg", "boundingBoxAnnotations": [
        {"displayName": "person", "xMin": 0.1, "yMin": 0.1, "xMax": 0.9, "yMax": 0.9}
    ]}

Run:
    GCP_PROJECT_ID=... GCS_ML_DATA_BUCKET=... python ml/training/automl/create_dataset.py
"""
import os
from google.cloud import aiplatform

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.getenv("GCP_REGION", "northamerica-northeast1")
GCS_BUCKET = os.environ["GCS_ML_DATA_BUCKET"]
IMPORT_FILE = os.getenv("AUTOML_IMPORT_FILE", "datasets/automl_import.jsonl")


def create_dataset() -> str:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    dataset = aiplatform.ImageDataset.create(
        display_name="homevision-person-detection",
        gcs_source=f"gs://{GCS_BUCKET}/{IMPORT_FILE}",
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.bounding_box,
        sync=True,
    )
    print(f"Dataset created: {dataset.resource_name}")
    return dataset.resource_name


if __name__ == "__main__":
    name = create_dataset()
    print(f"\nSet this as AUTOML_DATASET_RESOURCE_NAME={name}")
    print("Then run: python ml/training/automl/submit_training.py")
