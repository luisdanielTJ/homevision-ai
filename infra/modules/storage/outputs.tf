output "ml_data_bucket"         { value = google_storage_bucket.ml_data.name }
output "model_artifacts_bucket" { value = google_storage_bucket.model_artifacts.name }
