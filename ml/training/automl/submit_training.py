"""Submit AutoML Object Detection training job on Vertex AI.

Run:
    AUTOML_DATASET_RESOURCE_NAME=projects/.../datasets/... \\
    GCP_PROJECT_ID=... python ml/training/automl/submit_training.py
"""
import os
from google.cloud import aiplatform

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.getenv("GCP_REGION", "northamerica-northeast1")
DATASET_NAME = os.environ["AUTOML_DATASET_RESOURCE_NAME"]
MODEL_NAME = "homevision-automl-person-detector"


def submit() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    dataset = aiplatform.ImageDataset(DATASET_NAME)
    job = aiplatform.AutoMLImageTrainingJob(
        display_name=MODEL_NAME,
        prediction_type="object_detection",
        model_type="MOBILE_TF_LOW_LATENCY_1",  # fastest and cheapest
    )
    job.run(
        dataset=dataset,
        model_display_name=MODEL_NAME,
        budget_milli_node_hours=1000,  # 1 node-hour max
        training_fraction_split=0.7,
        validation_fraction_split=0.2,
        test_fraction_split=0.1,
        sync=False,
    )
    print(f"AutoML job submitted: {job.resource_name}")
    print("Monitor: Vertex AI Console → Training → Training pipelines")


if __name__ == "__main__":
    submit()
