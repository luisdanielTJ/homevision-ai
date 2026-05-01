from kfp.v2.dsl import component, Output, Dataset


@component(
    packages_to_install=["google-cloud-storage==2.17.0"],
    base_image="python:3.11-slim",
)
def data_ingestion_op(
    gcs_bucket: str,
    dataset_prefix: str,
    dataset: Output[Dataset],
) -> None:
    from google.cloud import storage
    from pathlib import Path

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    blobs = list(bucket.list_blobs(prefix=dataset_prefix))
    dest = Path(dataset.path)
    dest.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        rel = blob.name[len(dataset_prefix):].lstrip("/")
        local = dest / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
    print(f"Ingested {len(blobs)} files to {dest}")
