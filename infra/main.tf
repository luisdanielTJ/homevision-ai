terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  backend "gcs" {
    # Set via: terraform init -backend-config="bucket=YOUR_TFSTATE_BUCKET"
    prefix = "homevision-ai/tfstate"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

module "pubsub" {
  source     = "./modules/pubsub"
  project_id = var.project_id
}

module "storage" {
  source     = "./modules/storage"
  project_id = var.project_id
  region     = var.region
}

module "artifact_registry" {
  source     = "./modules/artifact_registry"
  project_id = var.project_id
  region     = var.region
}

module "firestore" {
  source     = "./modules/firestore"
  project_id = var.project_id
  region     = var.region
}
