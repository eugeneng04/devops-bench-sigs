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

"""Hermes CLI agent harness driving the ``hermes`` binary (local-only).

Capability wiring is delivered through hermes's native channels, laid down in
the run-scoped home directory ``$HERMES_HOME``:

* **State isolation** — ``HERMES_HOME`` points at ``<workspace>/.hermes``, so
  ``config.yaml`` and the ``state.db`` session store are per-run and never
  touch the user's ``~/.hermes``. The eval harness snapshots the workspace
  before the run, so ``.hermes`` is copied into the run's ``generated_files/``
  as a single directory rather than scattering hermes state across the diff.
* **MCP servers** — command-bearing bindings become ``mcp_servers`` entries in
  ``$HERMES_HOME/config.yaml``.
* **Skills** — ``config.capabilities.skills.paths`` are materialized under
  ``$HERMES_HOME/skills/<name>/SKILL.md``.
* **Rules** — ``config.capabilities.rules.text`` is prepended to the prompt
  (``hermes chat`` has no dedicated system-prompt flag).
* **Model auth** — ``config.api_key`` is threaded into the provider env vars
  the resolved :class:`~devops_bench.core.model_providers.ProviderSpec` names.

Trajectory and token usage are read back from the run's SQLite ``state.db``;
the parsers live in :mod:`~devops_bench.agents.cli.hermes.parsing`.

``__init__`` assigns ``self.rules``, ``self.mcp_servers`` and ``self.skills``
from the granted bindings, so the agent structurally satisfies
``SupportsRules`` / ``SupportsMcp`` / ``SupportsSkills``.
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from ruamel.yaml import YAML, YAMLError

from devops_bench.agents.base import AGENTS, AgentHarness
from devops_bench.agents.cli.hermes.parsing import (
    extract_tokens_from_db,
    extract_trajectory_from_db,
)
from devops_bench.agents.config import AgentConfig
from devops_bench.agents.result import AgentResult, empty_tokens
from devops_bench.agents.shared.cli_capabilities import (
    agent_workdir,
    build_mcp_servers,
    materialize_skills,
)
from devops_bench.core import ConfigError, SubprocessError, get_logger
from devops_bench.core.model_providers import resolve_provider
from devops_bench.core.subprocess import run

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from devops_bench.agents.capabilities import McpBinding

__all__ = ["HermesAgent"]

_log = get_logger("agents.cli.hermes.agent")

_HERMES_HOME_DIRNAME = ".hermes"
_CONFIG_FILE = "config.yaml"
_STATE_DB = "state.db"
_SOUL_FILE = "SOUL.md"
_SKILLS_DIRNAME = "skills"

# ruamel's safe round-trip: block style keeps the generated config readable if a
# run is inspected after the fact.
_yaml = YAML(typ="safe")
_yaml.default_flow_style = False


# Canonical bench provider -> the name hermes's ``--provider`` accepts. Not
# derivable from ``ProviderSpec``: hermes's ``vertex`` is Google-Vertex-Gemini
# only, and its OpenAI transport is spelled ``openai-api``, so mapping through
# ``adapter_family`` silently misroutes Claude-on-Vertex and hard-fails OpenAI.
_HERMES_PROVIDERS: dict[str, str] = {
    "google": "gemini",
    "google-vertex": "vertex",
    "anthropic": "anthropic",
    "anthropic-bedrock": "bedrock",
    "openai": "openai-api",
    "ollama": "ollama",
}

# Run-isolation env vars forwarded into every MCP server's ``env`` block. Hermes
# filters an MCP subprocess's environment down to an allowlist (PATH/HOME/... )
# plus whatever the server config names, so a per-run KUBECONFIG/CLOUDSDK_CONFIG
# would otherwise be dropped and the server would read the developer's ambient
# cluster and cloud config instead of the run's.
_MCP_ISOLATION_ENVS: tuple[str, ...] = ("KUBECONFIG", "CLOUDSDK_CONFIG")


def _hermes_provider(provider: str) -> str:
    """Map a bench provider alias onto the name ``hermes chat --provider`` takes.

    Args:
        provider: Raw ``AGENT_PROVIDER`` value.

    Returns:
        The hermes provider name.

    Raises:
        ConfigError: If hermes has no equivalent for the resolved provider.
    """
    canonical = resolve_provider(provider).canonical
    name = _HERMES_PROVIDERS.get(canonical)
    if name is None:
        # anthropic-vertex today: hermes's "vertex" serves Gemini only, so there
        # is no way to ask it for Claude on Vertex. Fail loudly rather than let
        # the run silently answer with a Gemini model.
        raise ConfigError(
            f"provider {canonical!r} has no hermes equivalent; "
            f"supported: {', '.join(sorted(_HERMES_PROVIDERS))}"
        )
    return name


def _prepend_rules(rules_text: str, prompt: str) -> str:
    """Return ``prompt`` with the granted rules prepended as a system brief."""
    if not rules_text.strip():
        return prompt
    return f"{rules_text.rstrip()}\n\n{prompt}"


def _build_env(config: AgentConfig) -> dict[str, str]:
    """Build the subprocess env overlay carrying provider credentials."""
    overlay: dict[str, str] = {}
    if config.provider and config.api_key:
        for var in resolve_provider(config.provider).api_key_envs:
            overlay[var] = config.api_key
    overlay.update(config.extra_env)
    return overlay


@AGENTS.register("hermes")
class HermesAgent(AgentHarness):
    """Hermes CLI agent harness driving the local ``hermes`` binary.

    Args:
        config: Agent configuration; ``target`` overrides the binary path.
        inherit_user_config: When ``True``, seed the run-scoped home from the
            user's ``~/.hermes`` (``config.yaml`` / ``SOUL.md``). Off by default
            so a benchmark run is reproducible and never picks up ambient
            developer state. ``.env`` is deliberately never inherited: it holds
            the user's provider credentials and the run home is copied into the
            run's artifacts.
    """

    def __init__(
        self, config: AgentConfig | None = None, *, inherit_user_config: bool = False
    ) -> None:
        AgentHarness.__init__(self, config)
        self.inherit_user_config = inherit_user_config
        caps = self.config.capabilities
        self.rules = caps.rules
        self.mcp_servers = caps.mcp_servers
        self.skills = caps.skills

    def _resolve_hermes_bin(self) -> str:
        """Resolve the ``hermes`` binary path, preferring an explicit target."""
        if self.config.target:
            return os.path.expanduser(self.config.target)
        candidate = os.path.expanduser("~/.local/bin/hermes")
        return candidate if os.path.exists(candidate) else "hermes"

    def _prepare_config(self, run_dir: Path, mcp_servers: tuple[McpBinding, ...]) -> None:
        """Write the run-scoped ``config.yaml``, merging in the MCP servers."""
        if self.inherit_user_config:
            user_dir = Path(os.path.expanduser("~/.hermes"))
            for name in (_CONFIG_FILE, _SOUL_FILE):
                if (user_dir / name).exists():
                    shutil.copy(user_dir / name, run_dir / name)

        config_path = run_dir / _CONFIG_FILE
        config_data: dict = {}
        if config_path.exists():
            try:
                loaded = _yaml.load(config_path.read_text(encoding="utf-8"))
            except (OSError, YAMLError) as exc:
                _log.warning("Failed to load existing %s: %s", _CONFIG_FILE, exc)
            else:
                # A seeded config holding a list or scalar parses fine but would
                # blow up the merge below; hermes would reject it anyway.
                if isinstance(loaded, dict):
                    config_data = loaded
                elif loaded is not None:
                    _log.warning(
                        "Ignoring existing %s: expected a mapping, got %s",
                        _CONFIG_FILE,
                        type(loaded).__name__,
                    )

        servers = build_mcp_servers(mcp_servers)
        if servers:
            # MCP servers are spawned by hermes, so they inherit its env rather
            # than the harness's; pass the run's isolation vars through
            # explicitly or the servers fall back to the ambient ones.
            for key in _MCP_ISOLATION_ENVS:
                value = os.environ.get(key)
                if value:
                    for entry in servers.values():
                        entry.setdefault("env", {})[key] = value
            config_data["mcp_servers"] = {**(config_data.get("mcp_servers") or {}), **servers}

        buffer = io.StringIO()
        _yaml.dump(config_data, buffer)
        config_path.write_text(buffer.getvalue(), encoding="utf-8")

    def _build_command(self, prompt: str) -> list[str]:
        """Build the ``hermes chat`` argv for this run.

        Raises:
            ConfigError: If the resolved provider has no hermes equivalent.
        """
        # ``--query=<prompt>`` as one token: the prompt is attacker-influenced
        # task text, and a separate argv element starting with ``-`` would be
        # parsed as a flag.
        cmd = [self._resolve_hermes_bin(), "chat", f"--query={prompt}"]
        if self.config.model:
            cmd.extend(["-m", self.config.model])
        if self.config.provider:
            cmd.extend(["--provider", _hermes_provider(self.config.provider)])
        return cmd

    def _execute(self, prompt: str, workspace_path: Path | None = None) -> AgentResult:
        caps = self.config.capabilities

        with agent_workdir(workspace_path, prefix="hermes-run-") as workdir:
            hermes_home = workdir / _HERMES_HOME_DIRNAME
            hermes_home.mkdir(parents=True, exist_ok=True)
            self._prepare_config(hermes_home, caps.mcp_servers)
            if caps.skills.paths:
                materialize_skills(hermes_home / _SKILLS_DIRNAME, caps.skills.paths)

            env_overlay = _build_env(self.config)
            env_overlay["HERMES_HOME"] = str(hermes_home)
            db_path = hermes_home / _STATE_DB

            try:
                completed = run(
                    self._build_command(_prepend_rules(caps.rules.text, prompt)),
                    check=False,
                    cwd=str(workdir),
                    timeout=self.config.timeout_sec,
                    extra_env=env_overlay,
                )
            except SubprocessError as exc:
                # With check=False the only SubprocessError left is the timeout.
                # The killed run still flushed whatever it completed, so report
                # that partial trajectory rather than discarding the run.
                trajectory, errors = extract_trajectory_from_db(db_path)
                return AgentResult(
                    output=(
                        f"Timeout expired.\n\n=== STDOUT ===\n{exc.stdout or ''}"
                        f"\n\n=== STDERR ===\n{exc.stderr or ''}"
                    ),
                    trajectory=trajectory,
                    tokens=extract_tokens_from_db(db_path),
                    errors=[f"hermes agent timed out after {self.config.timeout_sec:g}s", *errors],
                    metadata={"timeout": True},
                )
            except OSError as exc:
                # Binary missing/not executable: same canonical all-None token
                # shape as every other path, so the row reads "unavailable".
                return AgentResult(
                    output=f"Error: hermes binary unavailable: {exc}",
                    trajectory=[],
                    tokens=empty_tokens(),
                    errors=[f"hermes binary unavailable: {exc}"],
                )

            errors: list[str] = []
            metadata: dict = {}
            if completed.returncode != 0:
                stderr = (completed.stderr or "").strip()
                errors.append(
                    f"hermes agent exited {completed.returncode}: {stderr or '<no stderr>'}"
                )
                metadata["returncode"] = completed.returncode

            trajectory, export_errors = extract_trajectory_from_db(db_path)
            errors.extend(export_errors)
            tokens = extract_tokens_from_db(db_path)
            if completed.returncode == 0 and not tokens["total"]:
                # hermes exits 0 on an unknown provider, a rejected API key, or
                # an API call whose retries all failed. Without this the run
                # looks like a genuinely bad answer instead of an infra failure,
                # because no model usage was ever recorded.
                stderr = (completed.stderr or "").strip()
                errors.append(
                    "hermes agent recorded no model usage despite exiting 0; "
                    f"the model was never reached: {stderr or '<no stderr>'}"
                )

        return AgentResult(
            output=completed.stdout or "",
            trajectory=trajectory,
            tokens=tokens,
            errors=errors,
            metadata=metadata,
        )
