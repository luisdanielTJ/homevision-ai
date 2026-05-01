.PHONY: dev test lint infra-init infra-plan infra-apply endpoints-off deploy

dev:
	docker compose up --build

test:
	pytest services/ ml/ -v

lint:
	ruff check services/ ml/ dashboard/

infra-init:
	cd infra && terraform init

infra-plan:
	cd infra && terraform plan -var-file=environments/dev/terraform.tfvars

infra-apply:
	cd infra && terraform apply -var-file=environments/dev/terraform.tfvars

endpoints-off:
	gcloud ai endpoints undeploy-model $$VERTEX_AUTOML_ENDPOINT_ID \
		--region=$$GCP_REGION --project=$$GCP_PROJECT_ID || true
	gcloud ai endpoints undeploy-model $$VERTEX_YOLOV8_ENDPOINT_ID \
		--region=$$GCP_REGION --project=$$GCP_PROJECT_ID || true

deploy:
	gcloud builds submit --config=cloudbuild.yaml .
