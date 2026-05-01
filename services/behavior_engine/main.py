import os
import uvicorn
from dotenv import load_dotenv
from behavior_engine.dwell_tracker import DwellTracker
from behavior_engine.alert_engine import AlertEngine
from behavior_engine.gemini_enricher import GeminiEnricher
from behavior_engine.app import create_app

load_dotenv()

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.getenv("GCP_REGION", "northamerica-northeast1")
MOCK = os.getenv("MOCK_GEMINI", "false").lower() == "true"

app = create_app(
    dwell_tracker=DwellTracker(),
    alert_engine=AlertEngine(
        dwell_threshold=float(os.getenv("DWELL_THRESHOLD_SECONDS", "15")),
        head_turn_threshold=int(os.getenv("HEAD_TURN_THRESHOLD", "2")),
    ),
    enricher=GeminiEnricher(project_id=PROJECT_ID, region=REGION, mock=MOCK),
    alerts_topic=f"projects/{PROJECT_ID}/topics/{os.environ['PUBSUB_ALERTS_TOPIC']}",
    mock_pubsub=MOCK,
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)
