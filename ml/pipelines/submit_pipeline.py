"""Compile and submit the HomeVision AI training pipeline to Vertex AI Pipelines."""
import os
import json
from pathlib import Path

from kfp.v2 import compiler
from google.cloud import aiplatform

from pipeline import homevision_pipeline, PIPELINE_NAME

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.getenv("GCP_REGION", "northamerica-northeast1")
GCS_BUCKET = os.environ["GCS_BUCKET"]
DATASET_PREFIX = os.getenv("DATASET_PREFIX", "dataset/")
AUTOML_DATASET_NAME = os.environ["AUTOML_DATASET_RESOURCE_NAME"]
YOLOV8_IMAGE_URI = os.environ["YOLOV8_IMAGE_URI"]
PIPELINE_ROOT = f"gs://{GCS_BUCKET}/pipeline-root"
MONITORING_BUCKET = os.getenv("MONITORING_BUCKET", GCS_BUCKET)

COMPILED_PIPELINE_PATH = Path(__file__).parent / "homevision_pipeline.json"


def compile_pipeline() -> None:
    compiler.Compiler().compile(
        pipeline_func=homevision_pipeline,
        package_path=str(COMPILED_PIPELINE_PATH),
    )
    print(f"Pipeline compiled to {COMPILED_PIPELINE_PATH}")


def submit_pipeline() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.PipelineJob(
        display_name=PIPELINE_NAME,
        template_path=str(COMPILED_PIPELINE_PATH),
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "project_id": PROJECT_ID,
            "region": REGION,
            "gcs_bucket": GCS_BUCKET,
            "dataset_prefix": DATASET_PREFIX,
            "automl_dataset_name": AUTOML_DATASET_NAME,
            "yolov8_image_uri": YOLOV8_IMAGE_URI,
            "pipeline_root": PIPELINE_ROOT,
            "endpoint_display_name": "homevision-endpoint",
            "monitoring_bucket": MONITORING_BUCKET,
        },
        enable_caching=True,
    )
    job.submit()
    print(f"Pipeline submitted: {job.resource_name}")
    print("Monitor at: Vertex AI Console → Pipelines → Pipeline runs")


if __name__ == "__main__":
    compile_pipeline()
    submit_pipeline()
