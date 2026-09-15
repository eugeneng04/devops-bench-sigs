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

"""GCP provider: identity, GKE cluster access, and stack variable defaults."""

from __future__ import annotations

from typing import Any

from devops_bench.core import (
    ClusterInfo,
    ConfigError,
    NetworkPlan,
    SandboxError,
    get_bool,
    get_env,
    get_logger,
)
from devops_bench.core.subprocess import run
from devops_bench.providers.base import PROVIDERS, Provider, ResolveContext

__all__ = ["GcpProvider"]

_log = get_logger("providers.gcp")


def _context_name(project: str, location: str, cluster_name: str) -> str:
    """Return the kubectl context name ``gcloud get-credentials`` writes.

    gcloud derives it from the cluster's own coordinates rather than letting
    the caller choose, so reconstructing it is the only way to name the
    context without re-reading the kubeconfig.
    """
    return f"gke_{project}_{location}_{cluster_name}"


@PROVIDERS.register("gcp")
class GcpProvider(Provider):
    """Provider for GCP-hosted (GKE) clusters."""

    def ensure_account_credentials(self) -> None:
        """Ensure GCP application-default credentials are active.

        Currently a no-op: runs assume ambient credentials (ADC, a service
        account key, or workload identity) configured out of band.
        """
        _log.debug("GCP provider: assuming ambient application-default credentials")

    def ensure_cluster_credentials(
        self,
        cluster_name: str,
        location: str,
        variables: dict[str, Any],
        outputs: dict[str, Any] | None = None,
    ) -> ClusterInfo:
        """Configure ``kubectl`` for a GKE cluster via ``gcloud``.

        Args:
            cluster_name: Cluster name from the stack outputs.
            location: Cloud region or zone from the stack outputs.
            variables: OpenTofu input variables the cluster was provisioned with.
            outputs: Optional OpenTofu output values from provisioning.

        Returns:
            The cluster's :class:`~devops_bench.core.ClusterInfo`.

        Raises:
            ConfigError: If no project is resolvable from ``variables`` or the
                ``GCP_PROJECT_ID`` environment variable.
        """
        project = variables.get("project_id") or get_env("GCP_PROJECT_ID")
        if not project:
            raise ConfigError("Project ID not found in variables or environment (GCP_PROJECT_ID).")

        _log.info("Configuring kubectl for cluster: %s in %s...", cluster_name, location)
        run(
            [
                "gcloud",
                "container",
                "clusters",
                "get-credentials",
                cluster_name,
                "--location",
                location,
                "--project",
                project,
            ],
            capture=False,
        )

        context_name = _context_name(project, location, cluster_name)
        if get_bool("GCP_USE_ADC", False):
            _log.info(
                "Enabling application default credentials for auth plugin in context %s",
                context_name,
            )
            run(
                [
                    "kubectl",
                    "config",
                    "set-credentials",
                    context_name,
                    "--exec-arg=--use_application_default_credentials",
                ],
                capture=False,
            )
        else:
            _log.info(
                "Skipping GKE ADC credentials override (GCP_USE_ADC is false) for context %s",
                context_name,
            )

        return ClusterInfo.from_dict(
            {
                "name": cluster_name,
                "location": location,
                "project": project,
                # A stack whose task needs its own cloud API calls (beyond
                # kubectl) provisions a run-unique service account for them
                # and names it in this output; most stacks don't, and the
                # field stays None.
                "agent_cloud_identity": (outputs or {}).get("agent_cloud_identity"),
            }
        )

    def sandbox_network_plan(self, cluster_info: ClusterInfo) -> NetworkPlan:
        """Reach a GKE apiserver from inside a container over ordinary routing.

        A GKE endpoint is a real address — public, or VPC-routable from the
        bastion — so a bridge-networked container reaches it with no network
        or URL surgery. All this adds is the context pin, so the credential
        the container is handed is definitely for *this* cluster and not
        whichever GKE context the operator's kubeconfig last selected.

        Args:
            cluster_info: The provisioned cluster to reach.

        Returns:
            A default plan pinned to this cluster's context, or an unpinned
            one when the cluster's project or location is unknown — a wrong
            pin would be worse than none, since the sandbox refuses a context
            kubectl does not know.
        """
        if not (cluster_info.project and cluster_info.location):
            _log.warning(
                "cluster %s has no project/location; the sandbox kubeconfig will be "
                "built from the ambient current-context",
                cluster_info.name,
            )
            return NetworkPlan()
        return NetworkPlan(
            kubectl_context=_context_name(
                cluster_info.project, cluster_info.location, cluster_info.name
            )
        )

    def sandbox_cloud_credential_env(self, cluster_info: ClusterInfo) -> dict[str, str]:
        """Mint a short-lived impersonated token for the agent's cloud calls.

        The token is minted host-side by impersonating the run-unique service
        account the task's stack provisioned (which holds only that task's
        narrow roles), and crosses the boundary by value. No key file exists
        and none is mounted; the operator's ambient ADC never crosses. The
        provisioning identity needs ``roles/iam.serviceAccountTokenCreator``
        on that service account — the stack that creates the account grants
        it, so the whole loop tears down with the run.

        ``CLOUDSDK_AUTH_ACCESS_TOKEN`` is how gcloud accepts a bare token;
        ``GOOGLE_OAUTH_ACCESS_TOKEN`` covers the client libraries and tofu.
        Impersonated access tokens live at most one hour and are not
        refreshed inside the container, so an agent still making cloud calls
        past that gets a clean 401, not a silent widening.

        Args:
            cluster_info: The provisioned cluster; ``agent_cloud_identity``
                names the service account to impersonate, or None.

        Returns:
            The env to inject, or empty when the task declared no identity.

        Raises:
            SandboxError: When the identity is named but no token could be
                minted. Failing loud beats an agent that runs without the
                credential its task depends on.
        """
        identity = cluster_info.agent_cloud_identity
        if not identity:
            return {}
        result = run(
            [
                "gcloud",
                "auth",
                "print-access-token",
                f"--impersonate-service-account={identity}",
            ],
            check=False,
        )
        token = (result.stdout or "").strip()
        if result.returncode != 0 or not token:
            raise SandboxError(
                f"could not mint an access token for the agent's cloud identity "
                f"{identity!r} (gcloud exit {result.returncode}); the provisioning "
                "identity needs roles/iam.serviceAccountTokenCreator on it — "
                "refusing to run the agent without the credential its task needs"
            )
        _log.info("minted a short-lived cloud credential for the sandboxed agent as %s", identity)
        env = {
            "CLOUDSDK_AUTH_ACCESS_TOKEN": token,
            "GOOGLE_OAUTH_ACCESS_TOKEN": token,
        }
        if cluster_info.project:
            env["CLOUDSDK_CORE_PROJECT"] = cluster_info.project
            env["GOOGLE_CLOUD_PROJECT"] = cluster_info.project
        return env

    def cleanup(
        self,
        cluster_info: ClusterInfo,
        variables: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """No-op: GKE cluster cleanup is handled by stack teardown."""
        del success

    def resolve_variables(
        self, ctx: ResolveContext, custom_variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve default OpenTofu variables for GCP-based stacks.

        Returns:
            A new mapping with ``project_id``, ``cluster_name``, and ``location``
            filled in where not already set, plus ``namespace`` from the
            ``NAMESPACE`` environment variable when present.
        """
        variables = custom_variables.copy()
        variables.setdefault("infra_provider", "gcp")
        variables.setdefault("project_id", ctx.project_id)
        variables.setdefault("cluster_name", ctx.cluster_name)
        variables.setdefault("location", ctx.location)
        namespace = get_env("NAMESPACE")
        if namespace is not None:
            variables.setdefault("namespace", namespace)
        kubeconfig_path = get_env("KUBECONFIG")
        if kubeconfig_path:
            variables.setdefault("kubeconfig_path", kubeconfig_path)
        return variables
