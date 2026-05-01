# 02 — Build and Push Docker Images

## Prerequisites
- Docker installed and running
- Artifact Registry repo created (see 01-terraform-infra.md)
- `gcloud auth configure-docker northamerica-northeast1-docker.pkg.dev`

## Variables
```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="northamerica-northeast1"
export REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/homevision-ai"
export TAG="latest"
```

## Build each service

```bash
# Inference service
docker build -t ${REGISTRY}/inference_service:${TAG} services/inference_service/

# Behavior engine
docker build -t ${REGISTRY}/behavior_engine:${TAG} services/behavior_engine/

# Notification service
docker build -t ${REGISTRY}/notification_service:${TAG} services/notification_service/

# Dashboard
docker build -t ${REGISTRY}/dashboard:${TAG} dashboard/

# YOLOv8 training image
docker build -t ${REGISTRY}/yolov8_training:${TAG} ml/training/yolov8/
```

## Push
```bash
docker push ${REGISTRY}/inference_service:${TAG}
docker push ${REGISTRY}/behavior_engine:${TAG}
docker push ${REGISTRY}/notification_service:${TAG}
docker push ${REGISTRY}/dashboard:${TAG}
docker push ${REGISTRY}/yolov8_training:${TAG}
```

## Verify
Artifact Registry → Repositories → `homevision-ai` — all 5 images should appear with a recent `Updated` timestamp.
