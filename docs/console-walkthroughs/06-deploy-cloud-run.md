# 06 — Deploy Services to Cloud Run

Cloud Build handles this automatically on every push to `main` (see `cloudbuild.yaml`). These steps show how to deploy manually.

## Prerequisites
- Images pushed to Artifact Registry (see 02)
- Secrets stored in Secret Manager:
  - `GEMINI_API_KEY`
  - `GCP_PROJECT_ID`

## Deploy each service

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="northamerica-northeast1"
export REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/homevision-ai"
export TAG="latest"

# Inference service (private — called by Pub/Sub push)
gcloud run deploy inference-service \
  --image=${REGISTRY}/inference_service:${TAG} \
  --region=${REGION} \
  --no-allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}" \
  --set-secrets="VERTEX_ENDPOINT_ID=VERTEX_ENDPOINT_ID:latest"

# Behavior engine (private)
gcloud run deploy behavior-engine \
  --image=${REGISTRY}/behavior_engine:${TAG} \
  --region=${REGION} \
  --no-allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}" \
  --set-secrets="GEMINI_API_KEY=GEMINI_API_KEY:latest"

# Notification service (private — WebSocket + Pub/Sub push)
gcloud run deploy notification-service \
  --image=${REGISTRY}/notification_service:${TAG} \
  --region=${REGION} \
  --no-allow-unauthenticated \
  --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID}"

# Dashboard (public)
gcloud run deploy homevision-dashboard \
  --image=${REGISTRY}/dashboard:${TAG} \
  --region=${REGION} \
  --allow-unauthenticated \
  --set-env-vars="NOTIFICATION_SERVICE_URL=https://notification-service-....run.app"
```

## Wire up Pub/Sub push subscriptions

After services are deployed, update Terraform variables with their URLs and re-run `terraform apply`, or update manually:

```bash
# Get service URLs
INFERENCE_URL=$(gcloud run services describe inference-service --region=${REGION} --format='value(status.url)')
BEHAVIOR_URL=$(gcloud run services describe behavior-engine --region=${REGION} --format='value(status.url)')
NOTIFICATION_URL=$(gcloud run services describe notification-service --region=${REGION} --format='value(status.url)')

# Update Pub/Sub subscriptions
gcloud pubsub subscriptions modify-push-config raw-frames-sub \
  --push-endpoint="${INFERENCE_URL}/pubsub/push"
gcloud pubsub subscriptions modify-push-config detections-sub \
  --push-endpoint="${BEHAVIOR_URL}/pubsub/push"
gcloud pubsub subscriptions modify-push-config alerts-sub \
  --push-endpoint="${NOTIFICATION_URL}/pubsub/push"
```

## Verify

Cloud Run → Services — all 4 services showing green (last revision: serving 100% traffic).

Open the dashboard URL in a browser — you should see the 3-tab Streamlit UI.

## Stop endpoints to save credits

```bash
make endpoints-off
```
This scales all Cloud Run services to 0 replicas.
