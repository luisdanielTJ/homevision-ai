# 04 — YOLOv8 Custom Container Training on Vertex AI

## Prerequisites
- YOLOv8 training image pushed to Artifact Registry (see 02-build-push-images.md)
- Dataset in GCS under `dataset/` prefix (images + labels in YOLO format)
- Vertex AI API enabled

## Variables
```bash
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
export GCP_REGION="northamerica-northeast1"
export GCS_BUCKET="YOUR_ML_DATA_BUCKET"
export YOLOV8_IMAGE_URI="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/homevision-ai/yolov8_training:latest"
```

## Submit training job

The KFP pipeline handles this automatically (see 05-vertex-pipeline.md), but you can also run it manually:

```bash
python - <<'EOF'
from google.cloud import aiplatform
import os

aiplatform.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_REGION"])
job = aiplatform.CustomContainerTrainingJob(
    display_name="yolov8-homevision-manual",
    container_uri=os.environ["YOLOV8_IMAGE_URI"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest",
)
model = job.run(
    model_display_name="homevision-yolov8",
    machine_type="n1-standard-4",
    replica_count=1,
    environment_variables={
        "GCS_BUCKET": os.environ["GCS_BUCKET"],
        "DATASET_PREFIX": "dataset/",
        "EPOCHS": "30",
    },
    sync=True,
)
print(model.resource_name)
EOF
```

## Console verification

Vertex AI → Training → Custom jobs — job status: `SUCCEEDED`

Vertex AI → Model Registry → `homevision-yolov8` → Evaluate — check mAP50.

## Estimated cost
~$0.50–1.00 for 30 epochs on `n1-standard-4` with YOLOv8n (nano).
