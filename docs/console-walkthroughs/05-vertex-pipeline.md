# 05 — Run the Full Training Pipeline on Vertex AI Pipelines

The KFP pipeline chains: ingest → train AutoML + train YOLOv8 (parallel) → compare → register champion → deploy endpoint.

## Prerequisites
- Terraform infra applied (see 01)
- Both training images pushed (see 02)
- AutoML dataset created (see 03)

## Variables
```bash
export GCP_PROJECT_ID="YOUR_PROJECT_ID"
export GCP_REGION="northamerica-northeast1"
export GCS_BUCKET="YOUR_ML_DATA_BUCKET"
export AUTOML_DATASET_RESOURCE_NAME="projects/.../datasets/..."
export YOLOV8_IMAGE_URI="northamerica-northeast1-docker.pkg.dev/${GCP_PROJECT_ID}/homevision-ai/yolov8_training:latest"
```

## Compile and submit

```bash
cd ml/pipelines/
pip install kfp==2.7.0 google-cloud-aiplatform==1.57.0

python submit_pipeline.py
# Prints: Pipeline submitted: projects/.../pipelineJobs/...
```

## Console walkthrough

1. **Vertex AI → Pipelines → Pipeline runs** — find `homevision-training-pipeline`
2. Click the run to open the DAG view — nodes turn green as they complete
3. Click the `evaluate_compare_op` node → **Metrics** tab — see `automl_map50`, `yolov8_map50`, `champion`
4. After completion: **Vertex AI → Model Registry** — `homevision-champion` model registered
5. **Vertex AI → Online prediction → Endpoints** — `homevision-endpoint` with 1–3 replicas
6. **Vertex AI → Model Monitoring** — weekly skew detection job attached

## Estimated runtime
~70–90 minutes (AutoML dominates). Cost: ~$5–8 total.

## Re-running with caching
The pipeline uses `enable_caching=True`. Re-runs skip steps whose inputs haven't changed, cutting cost for iteration.
