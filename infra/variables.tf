variable "project_id" {
  type        = string
  description = "GCP project ID"
}

variable "region" {
  type    = string
  default = "northamerica-northeast1"
}

variable "inference_service_image" {
  type    = string
  default = ""
}

variable "behavior_engine_image" {
  type    = string
  default = ""
}

variable "notification_service_image" {
  type    = string
  default = ""
}

variable "dashboard_image" {
  type    = string
  default = ""
}
