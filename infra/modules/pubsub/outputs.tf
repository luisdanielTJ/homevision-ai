output "raw_frames_topic"  { value = google_pubsub_topic.raw_frames.id }
output "detections_topic"  { value = google_pubsub_topic.detections.id }
output "alerts_topic"      { value = google_pubsub_topic.alerts.id }
