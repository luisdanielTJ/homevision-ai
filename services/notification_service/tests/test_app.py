import json
import base64
from fastapi.testclient import TestClient
from notification_service.app import create_app
from notification_service.firestore_writer import FirestoreWriter


def _make_client() -> TestClient:
    return TestClient(create_app(writer=FirestoreWriter(mock=True)))


def test_health_returns_ok():
    assert _make_client().get("/health").json() == {"status": "ok"}


def test_alerts_endpoint_returns_list():
    resp = _make_client().get("/alerts")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_pubsub_push_returns_ok():
    alert = {"camera_id": "cam0", "timestamp": 1000.0, "dwell_time": 17.0, "alert_text": "Help!"}
    encoded = base64.b64encode(json.dumps(alert).encode()).decode()
    resp = _make_client().post("/pubsub/push", json={"message": {"data": encoded}})
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_pubsub_push_writes_to_firestore():
    from unittest.mock import MagicMock
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection

    client = TestClient(create_app(writer=FirestoreWriter(db=mock_db)))
    alert = {"camera_id": "cam0", "timestamp": 1000.0, "dwell_time": 17.0, "alert_text": "Help!"}
    encoded = base64.b64encode(json.dumps(alert).encode()).decode()
    client.post("/pubsub/push", json={"message": {"data": encoded}})

    mock_collection.add.assert_called_once_with(alert)
