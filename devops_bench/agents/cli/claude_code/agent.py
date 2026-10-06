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

"""Claude Code CLI agent harness driving the ``claude`` binary.

Runs ``claude -p --output-format stream-json --verbose`` and parses the stdout
event stream (:mod:`~.parsing`). Capabilities use the CLI's cwd channels:
``CLAUDE.md`` for rules, ``.claude/skills/`` for skills, and ``--mcp-config``
with ``--strict-mcp-config`` always set so a stray ``.mcp.json`` never leaks
tools into a baseline arm. Auth is env-driven; ``CLAUDE_CONFIG_DIR`` is a fresh
per-run dir so global state never races across concurrent runs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from functools import cache
from pathlib import Path

from devops_bench.agents.base import AGENTS, AgentHarness
from devops_bench.agents.cli.claude_code.parsing import parse_stream_json
from devops_bench.agents.config import AgentConfig
from devops_bench.agents.result import AgentResult
from devops_bench.agents.shared.cli_capabilities import (
    agent_workdir,
    build_mcp_servers,
    materialize_skills,
)
from devops_bench.core import SubprocessError, get_logger
from devops_bench.core.config import get_bool
from devops_bench.core.model_providers import resolve_provider
from devops_bench.core.subprocess import run

__all__ = ["ClaudeCodeAgent"]

# Auto-loaded from the cwd as the operator brief (native system-prompt analog).
_CLAUDE_RULES_FILE = "CLAUDE.md"
# Workspace config dir/files Claude Code reads from its cwd: ``skills/`` is the
# skill-discovery root and ``mcp-config.json`` is passed via ``--mcp-config``.
_CLAUDE_CONFIG_DIRNAME = ".claude"
_CLAUDE_SKILLS_DIR = "skills"
_CLAUDE_MCP_FILE = "mcp-config.json"
# Relocates Claude Code's mutable global state (see _claude_config_dir).
_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"

_log = get_logger("agents.cli.claude_code")

# Child stderr is unbounded and reaches the persisted record, so clip it everywhere.
_STDERR_TAIL_CHARS = 2000

# Under -p the CLI waits a few seconds on an open, silent stdin and logs a warning;
# an empty string closes the pipe at once.
_CLOSED_STDIN = ""


def _stderr_tail(stderr: str | None) -> str:
    """Stripped last :data:`_STDERR_TAIL_CHARS` characters of ``stderr``."""
    return (stderr or "").strip()[-_STDERR_TAIL_CHARS:]


# From this version ``--mcp-config`` under -p waits for servers before the first
# turn; older binaries start anyway and score an MCP arm with no tools attached.
_MCP_WAIT_MIN_VERSION = (2, 1, 221)
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


@cache
def _claude_version(target: str) -> tuple[int, int, int] | None:
    """Parse ``claude --version``, or ``None`` when it cannot be determined.

    Unreadable is not an error (``config.target`` may be a wrapper script).
    Cached per target, inconclusive probes included.
    """
    try:
        completed = run([target, "--version"], check=False, timeout=30)
    except (OSError, SubprocessError):
        return None
    match = _VERSION_RE.search(completed.stdout or "")
    if completed.returncode != 0 or not match:
        return None
    return (int(match[1]), int(match[2]), int(match[3]))


def _build_argv(
    target: str,
    prompt: str,
    *,
    model: str | None,
    max_turns: int | None,
    mcp_config_path: str | None,
) -> list[str]:
    """Build the ``claude`` headless invocation for ``prompt``.

    ``--verbose`` is required for ``stream-json`` under ``-p``;
    ``--dangerously-skip-permissions`` avoids confirmation prompts;
    ``--strict-mcp-config`` is always passed so a stray ``.mcp.json`` cannot
    grant tools to a baseline arm. The prompt trails ``--`` so a leading ``-``
    is not parsed as a flag.

    Args:
        target: Path to the ``claude`` binary (already user-expanded).
        prompt: Task prompt, passed as an argv value.
        model: Model id for ``--model``, or ``None`` for the CLI default.
        max_turns: Cap for ``--max-turns``; ``None`` or non-positive means unset.
        mcp_config_path: Absolute path to the MCP config document, or ``None``.
    """
    argv = [
        target,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--strict-mcp-config",
    ]
    if model:
        argv.extend(["--model", model])
    if max_turns is not None and max_turns > 0:
        argv.extend(["--max-turns", str(max_turns)])
    if mcp_config_path:
        argv.extend(["--mcp-config", mcp_config_path])
    argv.extend(["--", prompt])
    return argv


def _build_env(config: AgentConfig, *, config_dir: str | None) -> dict[str, str]:
    """Build the env overlay that makes the Claude Code run model-agnostic.

    ``config.api_key`` lands on the provider's key var(s); Vertex and Bedrock
    set their ``CLAUDE_CODE_USE_*`` switch and routing vars. The model flows
    through ``--model``, not here.

    Args:
        config: Resolved :class:`AgentConfig` for this run.
        config_dir: Per-run ``CLAUDE_CONFIG_DIR``, or ``None`` to keep the operator's.

    Raises:
        ConfigError: If ``config.provider`` is not a known provider.
    """
    # Resolve unconditionally so an unknown provider fails loud even keyless.
    spec = resolve_provider(config.provider, default="anthropic")
    overlay: dict[str, str] = {
        # Headless hygiene: no background telemetry/error traffic, no autoupdate.
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
    }
    # An empty key would clobber an ambient one through the env overlay.
    if config.api_key:
        for var in spec.api_key_envs:
            overlay[var] = config.api_key
    if spec.backend == "vertex":
        overlay["CLAUDE_CODE_USE_VERTEX"] = "1"
        project = os.environ.get("GCP_PROJECT_ID")
        if project:
            overlay["ANTHROPIC_VERTEX_PROJECT_ID"] = project
        # Repo var, then the CLI's native CLOUD_ML_REGION, then "global".
        region = os.environ.get("GCP_VERTEX_LOCATION") or os.environ.get("CLOUD_ML_REGION")
        overlay["CLOUD_ML_REGION"] = region or "global"
    elif spec.backend == "bedrock":
        overlay["CLAUDE_CODE_USE_BEDROCK"] = "1"
    if config_dir is not None:
        overlay[_CONFIG_DIR_ENV] = config_dir
    # Operator extra_env is applied last and wins (the override escape hatch).
    if config.extra_env:
        overlay.update(config.extra_env)
    return overlay


@contextlib.contextmanager
def _claude_config_dir() -> Iterator[str | None]:
    """Yield a fresh per-run ``CLAUDE_CONFIG_DIR``, or ``None`` if operator-set.

    A non-empty ambient value is the operator's escape hatch (a cached login) and
    is left alone; an empty one counts as unset. Under ``BENCH_PARALLEL`` the
    hatch is refused, since concurrent runs would share one mutable config dir.
    """
    if os.environ.get(_CONFIG_DIR_ENV):
        if not get_bool("BENCH_PARALLEL", False):
            yield None
            return
        _log.warning(
            "ignoring ambient %s under BENCH_PARALLEL: concurrent runs would share "
            "one mutable Claude config dir; using a per-run dir instead",
            _CONFIG_DIR_ENV,
        )
    # ignore_cleanup_errors: straggler lock files or MCP children must not turn a
    # completed run into an errored one.
    with tempfile.TemporaryDirectory(prefix="claude-config-", ignore_cleanup_errors=True) as tmpdir:
        yield tmpdir


@AGENTS.register("claude")
class ClaudeCodeAgent(AgentHarness):
    """Claude Code CLI agent harness driving the ``claude`` binary.

    The binary comes from ``config.target``, else ``claude`` on ``PATH``; model
    via ``--model``, key via the env overlay. ``__init__`` assigns
    ``mcp_servers``/``skills``/``rules`` so the agent satisfies the ``Supports*``
    protocols. Sandboxed runs are refused until the host paths are ported (see
    ``supports_sandbox``).
    """

    # Refused at preflight: argv[0], --mcp-config and CLAUDE_CONFIG_DIR still cross
    # in host spelling. TODO(sandbox): translate them like antigravity/openclaw, then flip.
    supports_sandbox = False

    def __init__(self, config: AgentConfig | None = None) -> None:
        AgentHarness.__init__(self, config)
        caps = self.config.capabilities
        self.mcp_servers = caps.mcp_servers
        self.skills = caps.skills
        self.rules = caps.rules

    def _execute(self, prompt: str, workspace_path: Path | None = None) -> AgentResult:
        """Build argv, run the CLI, and parse the stream-json output.

        Capabilities are laid down in the run directory first via the cwd
        channels described in the module docstring. The temp working directory
        (when ``workspace_path`` is ``None``) and the per-run
        ``CLAUDE_CONFIG_DIR`` are cleaned up on return; a harness-supplied
        ``workspace_path`` is left for the harness to collect.
        """
        caps = self.config.capabilities
        target = os.path.expanduser(self.config.target or "claude")
        rules_text = caps.rules.text

        with agent_workdir(workspace_path, prefix="claude-run-") as workdir:
            if rules_text:
                (workdir / _CLAUDE_RULES_FILE).write_text(rules_text, encoding="utf-8")

            claude_dir = workdir / _CLAUDE_CONFIG_DIRNAME
            materialize_skills(claude_dir / _CLAUDE_SKILLS_DIR, caps.skills.paths)

            mcp_config_path: str | None = None
            servers = build_mcp_servers(caps.mcp_servers)
            if servers:
                claude_dir.mkdir(parents=True, exist_ok=True)
                mcp_path = claude_dir / _CLAUDE_MCP_FILE
                mcp_path.write_text(json.dumps({"mcpServers": servers}, indent=2), encoding="utf-8")
                # A binding's argv can carry a credential and this file is collected.
                mcp_path.chmod(0o600)
                mcp_config_path = str(mcp_path)
                version = _claude_version(target)
                if version is not None and version < _MCP_WAIT_MIN_VERSION:
                    return AgentResult.errored(
                        "claude "
                        + ".".join(str(part) for part in version)
                        + " predates the --mcp-config startup wait (needs "
                        + ".".join(str(part) for part in _MCP_WAIT_MIN_VERSION)
                        + "); an MCP-bound arm would run without its servers. "
                        "Upgrade the binary or drop the mcp_servers binding."
                    )

            argv = _build_argv(
                target,
                prompt,
                model=self.config.model,
                max_turns=self.config.max_turns,
                mcp_config_path=mcp_config_path,
            )
            with _claude_config_dir() as config_dir:
                env_overlay = _build_env(self.config, config_dir=config_dir)
                try:
                    # Through the sandbox seam: containerised when config.sandbox is
                    # set, otherwise identical to the previous direct run(...).
                    completed = self.run_agent_cmd(
                        argv,
                        extra_env=env_overlay,
                        cwd=workdir,
                        check=False,
                        timeout=self.config.timeout_sec,
                        input=_CLOSED_STDIN,
                        host_run=run,
                    )
                except SubprocessError as exc:
                    # Under check=False this is a timeout: report it as one from the
                    # clipped stderr, and fall through to recover the partial stream.
                    stderr = _stderr_tail(exc.stderr)
                    returncode = exc.returncode
                    stdout = exc.stdout or ""
                    reason = f"claude timed out after {self.config.timeout_sec}s"
                    if stderr:
                        reason += f": {stderr}"
                except OSError as exc:
                    # Spawn failure: missing binary or vanished cwd.
                    return AgentResult.errored(f"failed to spawn claude: {exc}")
                else:
                    stderr = _stderr_tail(completed.stderr)
                    returncode = completed.returncode
                    stdout = completed.stdout or ""
                    reason = (
                        None
                        if returncode == 0
                        else f"claude exited {returncode}: {stderr or '<no stderr>'}"
                    )

        output, trajectory, tokens, parse_errors = parse_stream_json(stdout)
        errors: list[str] = list(parse_errors)
        metadata: dict = {}
        if stderr:
            # Kept on a clean exit too (e.g. MCP startup warnings).
            metadata["stderr"] = stderr
        if reason is not None:
            errors.append(reason)
            metadata["returncode"] = returncode
            output = output or f"Error: {reason}"
        return AgentResult(
            output=output,
            trajectory=trajectory,
            tokens=tokens,
            errors=errors,
            metadata=metadata,
        )
