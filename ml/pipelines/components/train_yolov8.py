from kfp.v2.dsl import component


@component(
    packages_to_install=["google-cloud-aiplatform==1.57.0"],
    base_image="python:3.11-slim",
)
def train_yolov8_op(
    gcs_bucket: str,
    dataset_prefix: str,
    project_id: str,
    region: str,
    yolov8_image_uri: str,
) -> str:
    """Submit YOLOv8 custom training job and return the model resource name."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    job = aiplatform.CustomContainerTrainingJob(
        display_name="yolov8-homevision-training",
        container_uri=yolov8_image_uri,
        model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest",
    )
    model = job.run(
        model_display_name="homevision-yolov8",
        machine_type="n1-standard-4",
        replica_count=1,
        environment_variables={
            "GCS_BUCKET": gcs_bucket,
            "DATASET_PREFIX": dataset_prefix,
            "EPOCHS": "30",
        },
        sync=True,
    )
    print(f"YOLOv8 model trained: {model.resource_name}")
    return model.resource_name
