# 01 — Provision Infrastructure with Terraform

## Prerequisites
- `gcloud auth application-default login`
- GCP project with billing enabled
- Terraform ≥ 1.6 installed

## Steps

### 1. Set variables
```bash
export TF_VAR_project_id="YOUR_PROJECT_ID"
export TF_VAR_region="northamerica-northeast1"
```

### 2. Initialise and apply
```bash
cd infra/
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

### 3. Verify in GCP Console
| Service | Console path |
|---------|-------------|
| Pub/Sub topics | Pub/Sub → Topics — expect `raw-frames`, `detections`, `alerts` |
| GCS buckets | Cloud Storage → Buckets — expect `ml-data-*` and `model-artifacts-*` |
| Artifact Registry | Artifact Registry → Repositories — expect `homevision-ai` |
| Firestore | Firestore → Data — expect Native-mode database |

### 4. Save outputs
```bash
terraform output -json > infra/outputs.json
```
The service `URL` values in this file are needed for Pub/Sub push subscriptions and Cloud Run deploy.
