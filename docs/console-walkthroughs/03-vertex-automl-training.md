# 03 — AutoML Object Detection Training on Vertex AI

## Prerequisites
- Dataset uploaded to GCS (see `ml/data/upload_to_gcs.py`)
- Vertex AI API enabled
- Service account with `roles/aiplatform.user`

## Step 1 — Create an Image Dataset in Vertex AI

```bash
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
export GCS_BUCKET="YOUR_ML_DATA_BUCKET"

python ml/automl/create_dataset.py
# Prints: Dataset resource name — copy it for next step
```

**Console verification:** Vertex AI → Datasets — your dataset should show row count > 0 and labels column = `person`.

## Step 2 — Submit AutoML training job

```bash
export AUTOML_DATASET_RESOURCE_NAME="projects/.../datasets/..."

python ml/training/automl/submit_training.py
# Prints AutoML job resource name
```

**Console path:** Vertex AI → Training → Training pipelines

The job takes ~1 node-hour (budget set to 1000 milli-node-hours). Status moves: `PENDING` → `RUNNING` → `SUCCEEDED`.

## Step 3 — Verify model metrics

Vertex AI → Model Registry → `homevision-automl-person-detector` → Evaluate

Key metric: **boundingBoxMeanAveragePrecision** — record this for the champion comparison step.

## Estimated cost
~$3–5 for 1000 milli-node-hours on `MOBILE_TF_LOW_LATENCY_1`.
