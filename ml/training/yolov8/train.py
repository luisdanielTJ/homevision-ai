"""YOLOv8n fine-tuning script for Vertex AI Custom Training Job.

Environment variables (set by Vertex AI):
    GCS_BUCKET       — GCS bucket containing the dataset
    DATASET_PREFIX   — GCS prefix for the YOLO dataset (default: datasets/yolo_person)
    AIP_MODEL_DIR    — Output directory for model artifacts (set by Vertex AI)
    EPOCHS           — Number of training epochs (default: 30)
    IMGSZ            — Input image size (default: 640)
"""
import os
import shutil
from pathlib import Path

from google.cloud import storage
from ultralytics import YOLO

GCS_BUCKET = os.environ["GCS_BUCKET"]
DATASET_PREFIX = os.environ.get("DATASET_PREFIX", "datasets/yolo_person")
MODEL_OUTPUT = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "640"))


def download_dataset(bucket_name: str, prefix: str, dest: Path) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    for blob in blobs:
        rel = blob.name[len(prefix):].lstrip("/")
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
    print(f"Downloaded {len(blobs)} files to {dest}")


def main() -> None:
    dest = Path("/tmp/dataset")
    download_dataset(GCS_BUCKET, DATASET_PREFIX, dest)

    model = YOLO("yolov8n.pt")
    results = model.train(
        data="/app/config.yaml",
        epochs=EPOCHS,
        imgsz=IMGSZ,
        device="cpu",
        project="/tmp/runs",
        name="yolov8_homevision",
        exist_ok=True,
    )
    map50 = results.results_dict.get("metrics/mAP50(B)", "N/A")
    print(f"Training complete. Best mAP@0.5: {map50}")

    best_weights = Path("/tmp/runs/yolov8_homevision/weights/best.pt")
    out_dir = Path(MODEL_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, out_dir / "best.pt")
    print(f"Model artifact saved to {out_dir}/best.pt")


if __name__ == "__main__":
    main()
