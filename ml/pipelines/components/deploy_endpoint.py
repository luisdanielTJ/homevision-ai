from kfp.v2.dsl import component


@component(
    packages_to_install=["google-cloud-aiplatform==1.57.0"],
    base_image="python:3.11-slim",
)
def deploy_endpoint_op(
    project_id: str,
    region: str,
    registered_model_name: str,
    endpoint_display_name: str,
    monitoring_bucket: str,
) -> str:
    """Deploy champion model to a Vertex AI endpoint with model monitoring enabled."""
    from google.cloud import aiplatform
    from google.cloud.aiplatform import model_monitoring

    aiplatform.init(project=project_id, location=region)

    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"',
        order_by="create_time desc",
    )
    endpoint = endpoints[0] if endpoints else aiplatform.Endpoint.create(
        display_name=endpoint_display_name
    )

    model = aiplatform.Model(registered_model_name)
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3,
        traffic_percentage=100,
    )

    skew_config = model_monitoring.TrainingPredictionSkewDetectionConfig(
        skew_thresholds={"confidence": 0.3},
    )
    monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name="homevision-monitoring",
        endpoint=endpoint,
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(sample_rate=0.1),
        monitor_interval=604800,  # 7 days in seconds
        alert_config=model_monitoring.EmailAlertConfig(
            user_emails=[],
        ),
        feature_attrib_settings=None,
        training_prediction_skew_config=skew_config,
        gcs_source=f"gs://{monitoring_bucket}/baseline/",
        data_format="jsonl",
    )
    print(f"Endpoint: {endpoint.resource_name}")
    print(f"Monitoring job: {monitoring_job.resource_name}")
    return endpoint.resource_name
