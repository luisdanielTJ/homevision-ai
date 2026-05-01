import json
import base64
from fastapi.testclient import TestClient
from inference_service.app import create_app
from inference_service.vertex_client import VertexInferenceClient


def _make_client() -> VertexInferenceClient:
    return VertexInferenceClient(
        project_id="test",
        region="us-central1",
        automl_endpoint_id="mock",
        yolov8_endpoint_id="mock",
        mock=True,
    )


def _make_test_client() -> TestClient:
    app = create_app(
        vertex_client=_make_client(),
        detections_topic="projects/p/topics/t",
        mock_pubsub=True,
    )
    return TestClient(app)


def test_health_returns_ok():
    client = _make_test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_pubsub_push_with_person_returns_200():
    client = _make_test_client()
    payload = json.dumps({
        "camera_id": "cam0",
        "timestamp": 1000.0,
        "keypoints": {
            "person_detected": True,
            "head_turn_count": 0,
            "object_raised": False,
            "nose_x": 0.5,
            "nose_y": 0.3,
            "left_wrist_y": 0.6,
            "right_wrist_y": 0.6,
            "left_shoulder_y": 0.4,
            "right_shoulder_y": 0.4,
        },
    })
    encoded = base64.b64encode(payload.encode()).decode()
    resp = client.post("/pubsub/push", json={"message": {"data": encoded}})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_pubsub_push_no_person_still_returns_200():
    """A frame with no person detected should not raise and should return 200."""
    client = _make_test_client()
    payload = json.dumps({
        "camera_id": "cam0",
        "timestamp": 1000.0,
        "keypoints": {
            "person_detected": False,
            "head_turn_count": 0,
            "object_raised": False,
            "nose_x": 0.0,
            "nose_y": 0.0,
            "left_wrist_y": 0.0,
            "right_wrist_y": 0.0,
            "left_shoulder_y": 0.0,
            "right_shoulder_y": 0.0,
        },
    })
    encoded = base64.b64encode(payload.encode()).decode()
    resp = client.post("/pubsub/push", json={"message": {"data": encoded}})
    assert resp.status_code == 200
