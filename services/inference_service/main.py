import os
import uvicorn
from dotenv import load_dotenv
from inference_service.vertex_client import VertexInferenceClient
from inference_service.app import create_app

load_dotenv()

MOCK = os.getenv("MOCK_VERTEX", "false").lower() == "true"

client = VertexInferenceClient(
    project_id=os.environ["GCP_PROJECT_ID"],
    region=os.getenv("GCP_REGION", "northamerica-northeast1"),
    automl_endpoint_id=os.environ["VERTEX_AUTOML_ENDPOINT_ID"],
    yolov8_endpoint_id=os.environ["VERTEX_YOLOV8_ENDPOINT_ID"],
    mock=MOCK,
)

app = create_app(
    vertex_client=client,
    detections_topic=(
        f"projects/{os.environ['GCP_PROJECT_ID']}"
        f"/topics/{os.environ['PUBSUB_DETECTIONS_TOPIC']}"
    ),
    mock_pubsub=MOCK,
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
