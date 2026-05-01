resource "google_artifact_registry_repository" "homevision" {
  project       = var.project_id
  location      = var.region
  repository_id = "homevision-ai"
  format        = "DOCKER"
  description   = "HomeVision AI Docker images"
}
