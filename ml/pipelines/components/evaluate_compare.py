from kfp.v2.dsl import component, Output, Metrics


@component(
    packages_to_install=["google-cloud-aiplatform==1.57.0"],
    base_image="python:3.11-slim",
)
def evaluate_compare_op(
    project_id: str,
    region: str,
    automl_model_name: str,
    yolov8_model_name: str,
    metrics: Output[Metrics],
) -> str:
    """Compare both models and return the champion model resource name."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)

    def get_map(model_name: str) -> float:
        model = aiplatform.Model(model_name)
        evals = model.list_model_evaluations()
        for ev in evals:
            val = ev.metrics.get("boundingBoxMeanAveragePrecision", None)
            if val is not None:
                return float(val)
        return 0.0

    automl_map = get_map(automl_model_name)
    yolov8_map = get_map(yolov8_model_name)

    metrics.log_metric("automl_map50", automl_map)
    metrics.log_metric("yolov8_map50", yolov8_map)

    champion = yolov8_model_name if yolov8_map >= automl_map else automl_model_name
    champion_name = "yolov8" if yolov8_map >= automl_map else "automl"
    metrics.log_metric("champion", champion_name)
    print(f"AutoML mAP: {automl_map:.3f} | YOLOv8 mAP: {yolov8_map:.3f} | Champion: {champion_name}")
    return champion
