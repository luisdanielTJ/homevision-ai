output "ml_data_bucket" {
  value = module.storage.ml_data_bucket
}

output "model_artifacts_bucket" {
  value = module.storage.model_artifacts_bucket
}

output "raw_frames_topic" {
  value = module.pubsub.raw_frames_topic
}

output "detections_topic" {
  value = module.pubsub.detections_topic
}

output "alerts_topic" {
  value = module.pubsub.alerts_topic
}
