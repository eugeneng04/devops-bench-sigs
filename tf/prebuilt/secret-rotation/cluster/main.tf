# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.0.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.0.0"
    }
  }
}

# Per-run random suffix so concurrent runs of the same task never collide on the
# project-global GCP resource names (service accounts, Secret Manager secrets).
# Created once per run and stable on re-apply (it lives in this run's state). The
# k8s namespace is intentionally NOT suffixed — it is cluster-scoped, so two
# separate clusters can both use the same namespace.
resource "random_id" "run" {
  byte_length = 4
}

# 1. GKE Cluster Provisioning
module "cluster" {
  source                   = "../../../modules/cluster"
  infra_provider           = "gcp"
  project_id               = var.project_id
  cluster_name             = var.cluster_name
  location                 = var.location
  node_count               = var.node_count
  machine_type             = var.machine_type
  enable_workload_identity = true
  # BYO-credentials model: the agent runs as the operator-provided broad runner
  # identity (the bastion VM SA), which is assumed to already hold the infra
  # perms it needs. So this stack does NOT grant it container.admin — the
  # module's project-level grant is disabled here (empty string -> count 0) to
  # avoid a shared binding that concurrent runs would fight over at teardown.
  agent_service_account = ""
  enable_iap_ssh        = true
}

# 3. GCP Secret Manager Setup
resource "google_secret_manager_secret" "db_credentials" {
  secret_id = "db-credentials-${var.namespace}-${random_id.run.hex}"
  project   = var.project_id
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "db_credentials_v1" {
  secret      = google_secret_manager_secret.db_credentials.id
  secret_data = "compromised-password-v1"
}

# 4. GCP IAM & GSA Configuration
resource "google_service_account" "secret_rotation_sa" {
  account_id   = "sa-${var.namespace}-${random_id.run.hex}"
  display_name = "GSA for GKE ExternalSecrets Secret Manager access"
  project      = var.project_id
}

resource "google_project_iam_member" "secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.secret_rotation_sa.email}"
}

resource "google_service_account_iam_member" "workload_identity" {
  service_account_id = google_service_account.secret_rotation_sa.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[external-secrets/external-secrets]"
}

# 5. Agent cloud identity for sandboxed runs.
#
# Rotating the credential is a Secret Manager write, and a sandboxed agent has
# no ambient cloud identity to make it with — by design. So the stack
# provisions the identity the agent's own cloud calls run as: a run-unique
# service account holding exactly the two roles the rotation needs, on exactly
# this run's secret. The harness impersonates it to mint a short-lived token
# for the container; the provisioning identity is granted tokenCreator on it
# here, an SA-level binding that tears down with the run's own SA — not a
# shared project-level grant for concurrent runs to fight over.
resource "google_service_account" "agent_rotator" {
  account_id   = "rot-${var.namespace}-${random_id.run.hex}"
  display_name = "Scoped identity for the sandboxed agent's Secret Manager calls"
  project      = var.project_id
}

resource "google_secret_manager_secret_iam_member" "agent_version_manager" {
  secret_id = google_secret_manager_secret.db_credentials.id
  role      = "roles/secretmanager.secretVersionManager"
  member    = "serviceAccount:${google_service_account.agent_rotator.email}"
}

resource "google_secret_manager_secret_iam_member" "agent_secret_accessor" {
  secret_id = google_secret_manager_secret.db_credentials.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_rotator.email}"
}

# Who may mint tokens for the rotator account. An explicit
# var.token_creator_member wins; otherwise the provisioner's own identity is
# derived from the ADC userinfo endpoint. That derivation is best-effort: a
# VM service-account credential without the userinfo-email scope reads a null
# email here, in which case the binding is skipped and the harness's mint
# fails loud, naming the missing grant — never a silent unsandboxed run.
data "google_client_openid_userinfo" "provisioner" {}

locals {
  provisioner_email = data.google_client_openid_userinfo.provisioner.email
  token_creator_member = (
    var.token_creator_member != "" ? var.token_creator_member
    : local.provisioner_email == null || local.provisioner_email == "" ? ""
    : endswith(local.provisioner_email, "gserviceaccount.com")
    ? "serviceAccount:${local.provisioner_email}"
    : "user:${local.provisioner_email}"
  )
}

resource "google_service_account_iam_member" "agent_rotator_token_creator" {
  count              = local.token_creator_member == "" ? 0 : 1
  service_account_id = google_service_account.agent_rotator.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = local.token_creator_member
}

# 10. Runner identity: BYO credentials.
#
# The agent runs as the operator-provided broad runner identity (the bastion VM
# SA, assumed pre-provisioned with the infra permissions it needs). This stack
# intentionally grants that identity NOTHING: adding per-run/project bindings to
# a shared SA is what caused concurrent runs to fight at teardown (one run's
# destroy revoked the binding the other still needed). A least-privilege,
# per-run runner SA is a tracked follow-up. The per-run resources above (the
# workload SA + secret) are name-suffixed via random_id so concurrent runs of
# this task never collide.
