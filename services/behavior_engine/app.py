import base64
import json
import time
from unittest.mock import MagicMock
from fastapi import FastAPI, Request
from google.cloud import pubsub_v1
from behavior_engine.dwell_tracker import DwellTracker
from behavior_engine.alert_engine import AlertEngine
from behavior_engine.gemini_enricher import GeminiEnricher


def create_app(
    dwell_tracker: DwellTracker,
    alert_engine: AlertEngine,
    enricher: GeminiEnricher,
    alerts_topic: str,
    mock_pubsub: bool = False,
) -> FastAPI:
    app = FastAPI(title="HomeVision Behavior Engine")
    publisher = MagicMock() if mock_pubsub else pubsub_v1.PublisherClient()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/pubsub/push")
    async def pubsub_push(request: Request):
        body = await request.json()
        data = base64.b64decode(body["message"]["data"]).decode()
        detection = json.loads(data)

        camera_id = detection["camera_id"]
        timestamp = detection["timestamp"]
        keypoints = detection["keypoints"]

        dwell_tracker.update(camera_id, timestamp)
        dwell_time = dwell_tracker.dwell_time(camera_id)
        head_turns = keypoints.get("head_turn_count", 0)
        object_raised = keypoints.get("object_raised", False)

        if alert_engine.should_alert(dwell_time, head_turns, object_raised):
            alert_text = enricher.enrich(camera_id, dwell_time, head_turns, object_raised)
            dwell_tracker.reset(camera_id)

            alert_msg = json.dumps({
                "camera_id": camera_id,
                "timestamp": time.time(),
                "dwell_time": dwell_time,
                "alert_text": alert_text,
                "automl": detection.get("automl", {}),
                "yolov8": detection.get("yolov8", {}),
            }).encode()

            if not mock_pubsub:
                future = publisher.publish(alerts_topic, alert_msg)
                future.result(timeout=5)

        return {"status": "ok"}

    return app
