import os
import uvicorn
from dotenv import load_dotenv
from behavior_engine.dwell_tracker import DwellTracker
from behavior_engine.alert_engine import AlertEngine
from behavior_engine.gemini_enricher import GeminiEnricher
from behavior_engine.app import create_app

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
REGION = os.getenv("GCP_REGION", "northamerica-northeast1")
MOCK_GEMINI = os.getenv("MOCK_GEMINI", "false").lower() == "true"
MOCK_PUBSUB = os.getenv("MOCK_PUBSUB", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8080"))

app = create_app(
    dwell_tracker=DwellTracker(),
    alert_engine=AlertEngine(
        dwell_threshold=float(os.getenv("DWELL_THRESHOLD_SECONDS", "15")),
        head_turn_threshold=int(os.getenv("HEAD_TURN_THRESHOLD", "2")),
    ),
    enricher=GeminiEnricher(
        project_id=PROJECT_ID,
        region=os.getenv("GEMINI_REGION", "us-central1"),
        mock=MOCK_GEMINI,
    ),
    alerts_topic=os.getenv(
        "ALERTS_TOPIC",
        f"projects/{PROJECT_ID}/topics/homevision-alerts",
    ),
    mock_pubsub=MOCK_PUBSUB,
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
