variable "project_id" { type = string }
variable "inference_service_url" {
  type    = string
  default = "https://placeholder.run.app/pubsub/push"
}
variable "behavior_engine_url" {
  type    = string
  default = "https://placeholder.run.app/pubsub/push"
}
variable "notification_service_url" {
  type    = string
  default = "https://placeholder.run.app/pubsub/push"
}
