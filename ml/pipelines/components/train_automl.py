from kfp.v2.dsl import component


@component(
    packages_to_install=["google-cloud-aiplatform==1.57.0"],
    base_image="python:3.11-slim",
)
def train_automl_op(
    project_id: str,
    region: str,
    automl_dataset_name: str,
) -> str:
    """Submit AutoML training and return model resource name."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    dataset = aiplatform.ImageDataset(automl_dataset_name)
    job = aiplatform.AutoMLImageTrainingJob(
        display_name="homevision-automl-person-detector",
        prediction_type="object_detection",
        model_type="MOBILE_TF_LOW_LATENCY_1",
    )
    model = job.run(
        dataset=dataset,
        model_display_name="homevision-automl",
        budget_milli_node_hours=1000,
        sync=True,
    )
    print(f"AutoML model trained: {model.resource_name}")
    return model.resource_name
