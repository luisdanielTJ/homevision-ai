resource "google_storage_bucket" "ml_data" {
  project                     = var.project_id
  name                        = "${var.project_id}-homevision-ml-data"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  lifecycle_rule {
    condition { age = 90 }
    action    { type = "Delete" }
  }
}

resource "google_storage_bucket" "model_artifacts" {
  project                     = var.project_id
  name                        = "${var.project_id}-homevision-models"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true
}
