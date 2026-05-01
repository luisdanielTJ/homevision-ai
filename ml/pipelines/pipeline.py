"""HomeVision AI Vertex AI Pipeline definition."""
from kfp.v2 import dsl
from kfp.v2.dsl import pipeline

from components.data_ingestion import data_ingestion_op
from components.train_automl import train_automl_op
from components.train_yolov8 import train_yolov8_op
from components.evaluate_compare import evaluate_compare_op
from components.register_champion import register_champion_op
from components.deploy_endpoint import deploy_endpoint_op

PIPELINE_NAME = "homevision-training-pipeline"


@pipeline(name=PIPELINE_NAME)
def homevision_pipeline(
    project_id: str,
    region: str,
    gcs_bucket: str,
    dataset_prefix: str,
    automl_dataset_name: str,
    yolov8_image_uri: str,
    pipeline_root: str,
    endpoint_display_name: str = "homevision-endpoint",
    monitoring_bucket: str = "",
):
    ingest = data_ingestion_op(
        gcs_bucket=gcs_bucket,
        dataset_prefix=dataset_prefix,
    )

    train_automl = train_automl_op(
        project_id=project_id,
        region=region,
        automl_dataset_name=automl_dataset_name,
    ).after(ingest)

    train_yolo = train_yolov8_op(
        gcs_bucket=gcs_bucket,
        dataset_prefix=dataset_prefix,
        project_id=project_id,
        region=region,
        yolov8_image_uri=yolov8_image_uri,
    ).after(ingest)

    compare = evaluate_compare_op(
        project_id=project_id,
        region=region,
        automl_model_name=train_automl.output,
        yolov8_model_name=train_yolo.output,
    ).after(train_automl, train_yolo)

    register = register_champion_op(
        project_id=project_id,
        region=region,
        champion_model_name=compare.output,
        pipeline_root=pipeline_root,
    ).after(compare)

    deploy_endpoint_op(
        project_id=project_id,
        region=region,
        registered_model_name=register.output,
        endpoint_display_name=endpoint_display_name,
        monitoring_bucket=monitoring_bucket,
    ).after(register)
