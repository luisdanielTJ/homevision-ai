# HomeVision AI — Design Spec
**Date:** 2026-04-30
**Status:** Approved

---

## 1. Problem Statement

Home Depot Canada employees use a mobile app to receive notifications when customers need help. Today, those notifications are triggered manually. This system automates detection: store cameras (or a webcam in demo mode) feed a real-time computer vision pipeline that identifies customers who appear to need assistance, enriches each alert with Gemini, and pushes a context-aware notification to a simulated employee dashboard.

**Primary business KPIs:**
- Reduce average customer wait time for assistance
- Increase associate response rate
- Improve in-store NPS

---

## 2. Scope

**In scope:**
- Real-time webcam frame ingestion via Cloud Pub/Sub
- Two deployed CV models on Vertex AI: AutoML baseline + fine-tuned YOLOv8
- Behavior analysis engine (dwell time, head turns, object hold-up)
- Gemini-powered alert enrichment
- Streamlit employee dashboard with live feed, alert history, and model comparison
- Full MLOps pipeline via Vertex AI Pipelines (KFP)
- Model monitoring (drift detection)
- Infrastructure as Code via Terraform
- CI/CD via Cloud Build triggered from GitHub
- GCP Console walkthrough documentation for each major step
- Local demo via laptop webcam

**Out of scope:**
- Production RTSP camera integration (future)
- Real Firebase FCM push to physical devices (simulated via WebSocket)
- Multi-store deployment (single dev environment only)
- Audio analysis

---

## 3. Architecture

### 3.1 Data Flow

```
Webcam (local laptop)
  └─► frame_publisher (Python client, 1–2 FPS)
          └─► MediaPipe pose/face extraction (local, free)
                  └─► Cloud Pub/Sub: raw-frames topic
                          └─► inference_service (Cloud Run)
                                  ├─► Vertex AI Endpoint A: AutoML
                                  └─► Vertex AI Endpoint B: YOLOv8
                                          └─► Cloud Pub/Sub: detections topic
                                                  └─► behavior_engine (Cloud Run)
                                                          ├─► Dwell time tracker (in-memory, per session)
                                                          ├─► Alert rule engine
                                                          ├─► Gemini API (alert enrichment)
                                                          └─► Cloud Pub/Sub: alerts topic
                                                                  └─► notification_service (Cloud Run)
                                                                          ├─► Firestore (alert history)
                                                                          └─► WebSocket → Streamlit Dashboard
```

### 3.2 Repository Structure

```
homevision-ai/
  ├── infra/                        # Terraform IaC
  │   ├── main.tf
  │   ├── variables.tf
  │   └── modules/
  │       ├── pubsub/
  │       ├── cloud_run/
  │       ├── vertex_ai/
  │       ├── firestore/
  │       └── storage/
  ├── services/
  │   ├── frame_publisher/          # Local Python client: webcam → Pub/Sub
  │   ├── inference_service/        # Cloud Run: CV inference against Vertex AI
  │   ├── behavior_engine/          # Cloud Run: dwell time logic + Gemini alerts
  │   └── notification_service/     # Cloud Run: Firestore + WebSocket
  ├── dashboard/                    # Streamlit app on Cloud Run
  ├── ml/
  │   ├── data/                     # Dataset prep scripts
  │   ├── training/                 # YOLOv8 fine-tuning + AutoML config
  │   └── pipelines/                # Vertex AI KFP pipeline definitions
  ├── docs/
  │   ├── superpowers/specs/        # This file and future specs
  │   └── console-walkthroughs/     # Step-by-step GCP Console guides
  └── .github/
      └── workflows/                # Cloud Build trigger configs
```

---

## 4. GCP Services

| Service | Role | Cost Control |
|---|---|---|
| Vertex AI | Training, endpoints, pipelines, experiments, model monitoring | Preemptible VMs for training; scale endpoints to 0 when idle |
| Cloud Run | All 4 services + dashboard | min-instances=0 on all services |
| Cloud Pub/Sub | Event bus (3 topics: raw-frames, detections, alerts) | Near zero at 1–2 FPS |
| Cloud Storage | Training data, model artifacts, frame snapshots | Standard storage, lifecycle rules |
| Firestore | Alert history, session metadata | Free tier sufficient |
| Artifact Registry | Docker images for Cloud Run services | Very low |
| Cloud Build | CI/CD on GitHub push | Free tier: 120 min/day |
| Secret Manager | Gemini API key, service account credentials | Free tier |
| Gemini API | Alert text enrichment (via Vertex AI SDK) | Called on alert trigger only, not per frame |
| Cloud Monitoring | Service health, latency dashboards | Free tier |

**Budget alert:** Set at $20 in GCP Billing to catch unexpected usage early.

---

## 5. Model Design

### 5.1 Model A — AutoML Baseline

- **Type:** Vertex AI AutoML Image Classification (person presence)
- **Training data:** ~200 labeled images uploaded via Vertex AI Dataset UI
- **Purpose:** Establish a baseline; demonstrates Console-first workflow
- **Deployment:** Vertex AI Endpoint, smallest machine type (n1-standard-2)

### 5.2 Model B — YOLOv8n Fine-tuned (Champion Candidate)

- **Type:** YOLOv8 nano (smallest variant, fastest inference, lowest cost)
- **Base weights:** COCO pretrained (`yolov8n.pt`)
- **Fine-tuning data:** COCO 2017 `person` subset + 50–100 custom webcam frames annotated in Label Studio (free, local)
- **Training:** Vertex AI Custom Training Job on preemptible `n1-standard-4` VM
- **Deployment:** Vertex AI Endpoint with custom container

### 5.3 Behavior Analysis Layer

Runs inside `behavior_engine` Cloud Run service (no extra GCP cost beyond the service itself). Stateful per camera session. MediaPipe keypoint extraction runs locally inside `frame_publisher` before frames are published to Pub/Sub:

| Signal | Detection method | Alert threshold |
|---|---|---|
| Dwell time | Frame count × FPS inverse | > 15 seconds |
| Head turns | MediaPipe face landmarks (local, pre-Pub/Sub) | > 2 turns in 10s window |
| Object raised | Wrist keypoint elevation above shoulder | Any occurrence |

Alert fires when: `dwell_time > 15s AND (head_turns > 2 OR object_raised == True)`

All thresholds configurable via environment variables.

### 5.4 Model Comparison

Both models evaluated on the same held-out test set. Metrics logged to Vertex AI Experiments:
- mAP@0.5
- Precision / Recall
- Inference latency (p50, p95)
- False positive rate

Champion model tagged in Vertex AI Model Registry and deployed as primary endpoint.

---

## 6. MLOps Pipeline

### 6.1 Vertex AI Pipeline (KFP)

Triggered manually from Console or CLI (not scheduled, to control costs):

```
data_ingestion_op
    └─► preprocessing_op
            └─► train_automl_op ──────────────────────────┐
            └─► train_yolov8_op (preemptible VM)           │
                                                           ▼
                                            evaluate_and_compare_op
                                                   └─► register_champion_op
                                                           └─► deploy_endpoint_op
```

### 6.2 Model Monitoring

- Enabled on champion endpoint only
- Prediction drift check: weekly batch (not continuous stream)
- Alert channel: Cloud Monitoring → email notification

### 6.3 Experiment Tracking

- Vertex AI Experiments used for all training runs
- Each run logs: hyperparameters, metrics, model artifact path
- Visible in GCP Console — shown during interview

---

## 7. CI/CD

- **Trigger:** Push to `main` branch on GitHub
- **Pipeline:** lint → unit tests → build Docker images → push to Artifact Registry → deploy to Cloud Run
- **Model retraining:** Manual only (not triggered by push) to control costs
- **Secrets:** Injected from Secret Manager at deploy time, never in source

---

## 8. Dashboard

Single Streamlit app deployed on Cloud Run with three tabs:

1. **Live Feed** — webcam stream with bounding boxes overlaid, current alert status
2. **Alert History** — table of past alerts from Firestore, with Gemini-generated text
3. **Model Comparison** — side-by-side metrics chart for AutoML vs YOLOv8

---

## 9. Demo Script (Interview)

1. Open Streamlit dashboard in browser
2. Start `frame_publisher` locally (`python services/frame_publisher/main.py`)
3. Walk into webcam frame, pause, look left and right twice
4. Within ~15 seconds: Gemini alert appears on dashboard
   - Example: *"Customer detected in zone A, dwell time 17s, appears to be comparing products."*
5. Switch to Model Comparison tab — walk through mAP and latency numbers
6. Open GCP Console: show Vertex AI endpoints, Pub/Sub topics, Cloud Run logs
7. Optional: show Vertex AI Pipeline run history and Experiments dashboard

---

## 10. Console Walkthrough Docs

| File | Content |
|---|---|
| `01-project-setup.md` | Billing, enable APIs, IAM service accounts |
| `02-vertex-ai-dataset.md` | Upload training data via Vertex AI Dataset UI |
| `03-automl-training.md` | Trigger AutoML job from Console |
| `04-vertex-ai-pipeline.md` | Run KFP pipeline from Console |
| `05-cloud-run-deploy.md` | Deploy Cloud Run services from Console |
| `06-monitoring.md` | Set up Model Monitoring + budget alert |

---

## 11. Local Development Setup

```bash
# One-command local stack (Docker Compose)
make dev

# Starts:
# - frame_publisher (connects to local webcam)
# - inference_service (mocked Vertex AI responses)
# - behavior_engine
# - notification_service
# - dashboard at localhost:8501
```

For GCP integration, `make deploy` runs Terraform + Cloud Build.

---

## 12. Credit Management Rules

Additional credits are acceptable to ensure a reliable, polished demo. The goal is efficiency, not restriction.

1. Budget alert at $50 in GCP Billing Console (enough headroom for training + demo runs)
2. All Cloud Run services: `min-instances=0` during development; bump to 1 for demo day
3. Vertex AI endpoints: shut down with `make endpoints-off` after development sessions; keep alive during demo prep
4. YOLOv8 training uses preemptible VMs to save cost, but fall back to standard VMs if job keeps failing
5. Frame ingestion at 2 FPS during dev; increase to 5 FPS for demo if latency feels sluggish
6. Gemini called on alert trigger only (estimated < 100 calls/day during dev)
7. Pub/Sub: no retained messages beyond 10 minutes
