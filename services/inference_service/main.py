import os
import uvicorn
from dotenv import load_dotenv
from inference_service.vertex_client import VertexInferenceClient
from inference_service.app import create_app

load_dotenv()

MOCK_VERTEX = os.getenv("MOCK_VERTEX", "false").lower() == "true"
MOCK_PUBSUB = os.getenv("MOCK_PUBSUB", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))

client = VertexInferenceClient(
    project_id=os.getenv("GCP_PROJECT_ID", ""),
    region=os.getenv("GCP_REGION", "northamerica-northeast1"),
    automl_endpoint_id=os.getenv("VERTEX_AUTOML_ENDPOINT_ID", ""),
    yolov8_endpoint_id=os.getenv("VERTEX_YOLOV8_ENDPOINT_ID", ""),
    mock=MOCK_VERTEX,
)

app = create_app(
    vertex_client=client,
    detections_topic=os.getenv(
        "DETECTIONS_TOPIC",
        f"projects/{os.getenv('GCP_PROJECT_ID', '')}/topics/homevision-detections",
    ),
    mock_pubsub=MOCK_PUBSUB,
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
