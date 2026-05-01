resource "google_pubsub_topic" "raw_frames" {
  project                    = var.project_id
  name                       = "homevision-raw-frames"
  message_retention_duration = "600s"
}

resource "google_pubsub_topic" "detections" {
  project                    = var.project_id
  name                       = "homevision-detections"
  message_retention_duration = "600s"
}

resource "google_pubsub_topic" "alerts" {
  project                    = var.project_id
  name                       = "homevision-alerts"
  message_retention_duration = "600s"
}

resource "google_pubsub_subscription" "inference_service_sub" {
  project              = var.project_id
  name                 = "homevision-raw-frames-inference-sub"
  topic                = google_pubsub_topic.raw_frames.id
  ack_deadline_seconds = 30

  push_config {
    push_endpoint = var.inference_service_url
  }
}

resource "google_pubsub_subscription" "behavior_engine_sub" {
  project              = var.project_id
  name                 = "homevision-detections-behavior-sub"
  topic                = google_pubsub_topic.detections.id
  ack_deadline_seconds = 30

  push_config {
    push_endpoint = var.behavior_engine_url
  }
}

resource "google_pubsub_subscription" "notification_service_sub" {
  project              = var.project_id
  name                 = "homevision-alerts-notification-sub"
  topic                = google_pubsub_topic.alerts.id
  ack_deadline_seconds = 30

  push_config {
    push_endpoint = var.notification_service_url
  }
}
