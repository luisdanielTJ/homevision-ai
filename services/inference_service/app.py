import base64
import json
from unittest.mock import MagicMock
from fastapi import FastAPI, Request
from google.cloud import pubsub_v1
from inference_service.vertex_client import VertexInferenceClient


def create_app(
    vertex_client: VertexInferenceClient,
    detections_topic: str,
    mock_pubsub: bool = False,
) -> FastAPI:
    app = FastAPI(title="HomeVision Inference Service")
    publisher = MagicMock() if mock_pubsub else pubsub_v1.PublisherClient()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/pubsub/push")
    async def pubsub_push(request: Request):
        body = await request.json()
        data = base64.b64decode(body["message"]["data"]).decode()
        frame = json.loads(data)
        keypoints = frame["keypoints"]

        automl_result = vertex_client.predict_automl(keypoints)
        yolov8_result = vertex_client.predict_yolov8(keypoints)

        detection_msg = json.dumps({
            "camera_id": frame["camera_id"],
            "timestamp": frame["timestamp"],
            "keypoints": keypoints,
            "automl": automl_result.to_dict(),
            "yolov8": yolov8_result.to_dict(),
        }).encode()

        if not mock_pubsub:
            future = publisher.publish(detections_topic, detection_msg)
            future.result(timeout=5)

        return {"status": "ok"}

    return app
