from kfp.v2.dsl import component


@component(
    packages_to_install=["google-cloud-aiplatform==1.57.0"],
    base_image="python:3.11-slim",
)
def register_champion_op(
    project_id: str,
    region: str,
    champion_model_name: str,
    pipeline_root: str,
) -> str:
    """Copy champion model to a versioned registry entry and return its resource name."""
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=region)
    source = aiplatform.Model(champion_model_name)
    registered = source.copy(
        destination_location=region,
        destination_project=project_id,
    )
    print(f"Registered champion: {registered.resource_name}")
    return registered.resource_name
