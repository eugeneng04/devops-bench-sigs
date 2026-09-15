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

"""Unit tests for devops_bench.agents.cli.openclaw."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from devops_bench.agents import AGENTS, AgentConfig
from devops_bench.agents.capabilities import (
    AgentRules,
    AllCapabilities,
    McpBinding,
    SkillBinding,
    SupportsMcp,
    SupportsRules,
    SupportsSkills,
)
from devops_bench.agents.cli.openclaw import OpenClawAgent, parse_trajectory_export
from devops_bench.agents.cli.openclaw import agent as oc_mod
from devops_bench.agents.cli.openclaw.agent import (
    _build_env,
    _build_local_command,
    _build_model_override,
    _build_openclaw_config,
    _oc_model_id,
)
from devops_bench.agents.cli.openclaw.parsing import _pick_session_key, _strip_ansi
from devops_bench.agents.sandbox import SandboxSpec
from devops_bench.core.errors import ConfigError, SubprocessError
from devops_bench.results.normalize import count_tool_calls


def _events(*entries: dict) -> str:
    return "\n".join(json.dumps(e) for e in entries) + "\n"


def _tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "tool.call",
        "data": {"toolCallId": call_id, "name": name, "arguments": arguments},
    }


def _tool_result(
    call_id: str, text: str, *, is_error: bool = False, status: str = "completed"
) -> dict:
    return {
        "type": "tool.result",
        "data": {
            "message": {
                "toolCallId": call_id,
                "content": [{"type": "text", "text": text}],
                "details": {"status": status},
                "isError": is_error,
            }
        },
    }


# Mirrors the real ``oc sessions export-trajectory`` events.jsonl schema:
# dotted ``type`` with a nested ``data`` payload (captured from oc 2026.6.9).
SAMPLE_EVENTS = _events(
    _tool_call("1", "kubectl_get_pods", {"namespace": "default"}),
    _tool_result("1", "pod-a Running\n"),
    _tool_call("2", "kubectl_describe", {"resource": "pod/pod-a"}),
    _tool_result("2", "Phase: Running"),
    {
        "type": "model.completed",
        "data": {
            "usage": {"input": 5, "output": 10, "total": 15},
            "assistantTexts": ["All pods healthy."],
        },
    },
)


def test_parse_trajectory_export_folds_call_result_pairs() -> None:
    export = parse_trajectory_export(SAMPLE_EVENTS)
    assert export.errors == []
    assert export.tokens == {"input": 5, "output": 10, "total": 15}
    assert export.output == "All pods healthy."
    assert export.trajectory == [
        {
            "name": "kubectl_get_pods",
            "args": {"namespace": "default"},
            "result": "pod-a Running\n",
            "status": "completed",
        },
        {
            "name": "kubectl_describe",
            "args": {"resource": "pod/pod-a"},
            "result": "Phase: Running",
            "status": "completed",
        },
    ]


def test_parse_trajectory_export_sums_usage_across_turns() -> None:
    """Token usage is summed across every model.completed, not just the last.

    OpenClaw reports usage per turn (per model call); a multi-turn session that
    kept only the final ``model.completed`` would undercount to a single call.
    """
    blob = _events(
        {"type": "model.completed", "data": {"usage": {"input": 100, "output": 20, "total": 120}}},
        _tool_call("1", "kubectl_get_pods", {}),
        _tool_result("1", "pod-a Running"),
        {"type": "model.completed", "data": {"usage": {"input": 250, "output": 30, "total": 280}}},
        {
            "type": "model.completed",
            "data": {"usage": {"input": 75, "output": 15, "total": 90}, "assistantTexts": ["done"]},
        },
    )
    export = parse_trajectory_export(blob)
    assert export.errors == []
    assert export.output == "done"
    assert export.tokens == {"input": 425, "output": 65, "total": 490}


def test_parse_trajectory_export_sums_cache_write_from_per_call_events() -> None:
    """``cacheWrite`` lives only on ``assistant.message``, not ``model.completed``.

    Shape taken from a live export: the per-call usages sum to the
    ``model.completed`` total for every bucket except ``cacheWrite``, which the
    rollup drops. Cache writes are priced above input on Anthropic, so losing
    the bucket understates what an openclaw run was billed.
    """
    blob = _events(
        {
            "type": "assistant.message",
            "data": {
                "message": {
                    "content": [],
                    "usage": {"input": 90, "output": 8, "cacheRead": 160, "cacheWrite": 12},
                }
            },
        },
        {
            "type": "assistant.message",
            "data": {
                "message": {
                    "content": [],
                    "usage": {"input": 92, "output": 11, "cacheRead": 162, "cacheWrite": 7},
                }
            },
        },
        {
            "type": "model.completed",
            "data": {
                "usage": {"input": 182, "output": 19, "cacheRead": 322, "total": 523},
                "assistantTexts": ["done"],
            },
        },
    )
    export = parse_trajectory_export(blob)
    assert export.errors == []
    assert export.tokens["cacheWrite"] == 19
    # The rollup total omits cacheWrite too, so it is folded back in: the
    # canonical contract is that total is the sum of every bucket.
    assert export.tokens["total"] == 182 + 19 + 322 + 19
    assert export.tokens["total"] == (
        export.tokens["input"]
        + export.tokens["output"]
        + export.tokens["cacheRead"]
        + export.tokens["cacheWrite"]
    )
    # The rollup buckets are untouched — nothing is counted from both sources.
    assert export.tokens["input"] == 182
    assert export.tokens["output"] == 19
    assert export.tokens["cacheRead"] == 322


def test_parse_trajectory_export_omits_cache_write_when_unreported() -> None:
    """No ``cacheWrite`` anywhere means the key is absent, so the row reads None.

    An older openclaw whose events carry no per-call usage must not be recorded
    as having written zero cache tokens.
    """
    blob = _events(
        {"type": "assistant.message", "data": {"message": {"content": []}}},
        {"type": "model.completed", "data": {"usage": {"input": 10, "output": 2}}},
    )
    assert "cacheWrite" not in parse_trajectory_export(blob).tokens


def test_parse_trajectory_export_counts_model_turns_not_tool_calls() -> None:
    """One ``assistant.message`` can issue several tool calls, so the two differ.

    Shape taken from a live openclaw run whose single message carried two
    ``toolCall`` entries. Reporting ``len(trajectory)`` as turns would say three
    round-trips happened where the provider was called twice, which is the
    number input tokens actually grow with.
    """
    blob = _events(
        {"type": "assistant.message", "data": {"message": {"content": []}}},
        _tool_call("1", "kubectl_get_pods", {}),
        _tool_result("1", "ok"),
        _tool_call("2", "kubectl_describe", {}),
        _tool_result("2", "ok"),
        {"type": "assistant.message", "data": {"message": {"content": []}}},
        {"type": "model.completed", "data": {"usage": {"input": 5}, "assistantTexts": ["done"]}},
    )
    export = parse_trajectory_export(blob)
    assert export.model_turns == 2
    assert len(export.trajectory) == 2


def test_parse_trajectory_export_leaves_model_turns_none_without_messages() -> None:
    """No ``assistant.message`` events means unmeasured, not zero turns.

    An export that produced output cannot have taken zero round-trips, so ``0``
    would be a parse miss dressed up as a fact and would drag a dashboard
    average down.
    """
    blob = _events(
        {"type": "model.completed", "data": {"usage": {"input": 5}, "assistantTexts": ["done"]}},
    )
    assert parse_trajectory_export(blob).model_turns is None


def test_parse_trajectory_export_records_every_model_that_answered() -> None:
    """openclaw fails over mid-run, so the requested model can be the wrong label.

    Two distinct ids across the run is the failover itself; both are kept, in
    first-seen order, and a repeat is not counted twice.
    """
    blob = _events(
        {"type": "assistant.message", "data": {"message": {"model": "gemini-3-flash-preview"}}},
        {"type": "assistant.message", "data": {"message": {"model": "gemini-3-flash-preview"}}},
        {"type": "assistant.message", "data": {"message": {"model": "claude-opus-4-5"}}},
    )
    export = parse_trajectory_export(blob)
    assert export.served_models == ["gemini-3-flash-preview", "claude-opus-4-5"]


def test_parse_trajectory_export_leaves_served_models_empty_when_unreported() -> None:
    blob = _events({"type": "assistant.message", "data": {"message": {"content": []}}})
    assert parse_trajectory_export(blob).served_models == []


def test_parse_trajectory_export_times_tools_and_merges_concurrent_calls() -> None:
    """Two calls issued in one message overlap, so their spans count once.

    Timestamps taken from a live export: both ``tool.call`` events landed on the
    same millisecond. Summing the two durations would say the run spent longer
    in tools than it ran for.
    """
    blob = _events(
        {**_tool_call("1", "a", {}), "ts": "2026-08-24T17:44:45.862Z"},
        {**_tool_call("2", "b", {}), "ts": "2026-08-24T17:44:45.862Z"},
        {**_tool_result("1", "ok"), "ts": "2026-08-24T17:44:46.081Z"},
        {**_tool_result("2", "ok"), "ts": "2026-08-24T17:44:46.362Z"},
    )
    assert parse_trajectory_export(blob).tool_wait_sec == pytest.approx(0.5, abs=1e-6)


def test_parse_trajectory_export_leaves_tool_wait_none_without_timestamps() -> None:
    """An export with no ``ts`` reports unmeasured, not zero seconds in tools."""
    blob = _events(_tool_call("1", "a", {}), _tool_result("1", "ok"))
    assert parse_trajectory_export(blob).tool_wait_sec is None


def test_parse_trajectory_export_coerces_a_null_tool_name() -> None:
    """A null name must not cost the call its row in the telemetry counts.

    ``dict.get(key, default)`` returns ``None`` when the key is present holding
    null, and ``count_tool_calls`` skips any entry whose name is not a ``str`` —
    so an export like this dropped a failed call and reported zero tool errors.
    """
    blob = _events(
        {"type": "tool.call", "data": {"toolCallId": "1", "name": None, "arguments": {}}},
        _tool_result("1", "boom", is_error=True),
    )
    export = parse_trajectory_export(blob)
    assert export.trajectory[0]["name"] == ""
    assert count_tool_calls(export.trajectory) == (1, 1)


def test_parse_trajectory_export_sums_nested_cost_breakdown() -> None:
    """Nested numeric mappings (e.g. a per-turn ``cost`` block) are summed too."""
    blob = _events(
        {"type": "model.completed", "data": {"usage": {"input": 10, "cost": {"total": 0.01}}}},
        {"type": "model.completed", "data": {"usage": {"input": 5, "cost": {"total": 0.02}}}},
    )
    export = parse_trajectory_export(blob)
    assert export.errors == []
    assert export.tokens["input"] == 15
    assert export.tokens["cost"]["total"] == pytest.approx(0.03)


def test_parse_trajectory_export_keeps_cache_write_inside_a_nested_cost_block() -> None:
    """The top-level ``cacheWrite`` handling must not reach into the cost breakdown.

    The token bucket is settled against the per-call events, but cost dollars
    have no second source -- dropping them would leave the itemized costs short
    of their own total.
    """
    blob = _events(
        {
            "type": "model.completed",
            "data": {
                "usage": {
                    "input": 10,
                    "cacheWrite": 4,
                    "cost": {"input": 0.5, "cacheWrite": 0.25, "total": 0.75},
                }
            },
        },
    )
    tokens = parse_trajectory_export(blob).tokens
    assert tokens["cost"] == {"input": 0.5, "cacheWrite": 0.25, "total": 0.75}


def test_parse_trajectory_export_does_not_double_count_a_reported_cache_write() -> None:
    """A rollup reporting ``cacheWrite`` already counted it in ``total``.

    Folding the per-call sum on top would report a total larger than the sum of
    its own buckets.
    """
    blob = _events(
        {
            "type": "model.completed",
            "data": {"usage": {"input": 10, "output": 5, "cacheWrite": 7, "total": 22}},
        },
        {"type": "assistant.message", "data": {"message": {"usage": {"cacheWrite": 7}}}},
    )
    tokens = parse_trajectory_export(blob).tokens
    assert tokens["cacheWrite"] == 7
    assert tokens["total"] == 22


def test_parse_trajectory_export_keeps_a_cache_write_only_the_rollup_reported() -> None:
    """The bucket must survive when no ``assistant.message`` carried it."""
    blob = _events(
        {
            "type": "model.completed",
            "data": {"usage": {"input": 10, "output": 5, "cacheWrite": 7, "total": 22}},
        },
        {"type": "assistant.message", "data": {"message": {"usage": {}}}},
    )
    tokens = parse_trajectory_export(blob).tokens
    assert tokens["cacheWrite"] == 7
    assert tokens["total"] == 22


def test_parse_trajectory_export_matches_reused_tool_ids_in_emission_order() -> None:
    """Two live calls can share an id; the second must not overwrite the first.

    Overwriting pairs the first call's result with the second call's start,
    reporting a tool wait shorter than the run and inventing an orphan error.
    """
    blob = _events(
        {**_tool_call("x", "a", {}), "ts": "2026-08-24T17:57:28.000Z"},
        {**_tool_call("x", "b", {}), "ts": "2026-08-24T17:57:29.000Z"},
        {**_tool_result("x", "ra"), "ts": "2026-08-24T17:57:30.000Z"},
        {**_tool_result("x", "rb"), "ts": "2026-08-24T17:57:32.000Z"},
    )
    export = parse_trajectory_export(blob)
    assert export.errors == []
    assert [(c["name"], c["result"]) for c in export.trajectory] == [("a", "ra"), ("b", "rb")]
    assert export.tool_wait_sec == pytest.approx(4.0, abs=1e-6)


def test_parse_trajectory_export_marks_failed_tool_result_as_error() -> None:
    """``isError`` (or ``details.status`` of error/failed) → status 'error'."""
    blob = _events(
        _tool_call("1", "exec", {"command": "false"}),
        _tool_result("1", "boom", is_error=True, status="error"),
    )
    export = parse_trajectory_export(blob)
    assert export.errors == []
    assert export.trajectory[0]["status"] == "error"


def test_parse_trajectory_export_output_falls_back_to_assistant_message() -> None:
    """When no model.completed.assistantTexts, output comes from assistant.message text."""
    blob = _events(
        {
            "type": "assistant.message",
            "data": {
                "message": {"role": "assistant", "content": [{"type": "text", "text": "done."}]}
            },
        },
        {"type": "model.completed", "data": {"usage": {"input": 1, "output": 2}}},
    )
    export = parse_trajectory_export(blob)
    assert export.tokens == {"input": 1, "output": 2}
    assert export.output == "done."


def test_parse_trajectory_export_placeholder_final_text_falls_back() -> None:
    """A redacted final answer engages the assistant.message fallback.

    The live shape from run_20260911_172304: oc's sanitizer stored
    ``[Malformed diagnostic JSON redacted]`` over the final message in its own
    sqlite before the harness read it. The placeholder must not become the
    graded output when the real text still rides on an assistant.message.
    """
    blob = _events(
        {
            "type": "assistant.message",
            "data": {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Deployed hello-app; 3/3 pods Ready."}],
                }
            },
        },
        {
            "type": "model.completed",
            "data": {
                "usage": {"input": 1, "output": 2},
                "assistantTexts": ["[Malformed diagnostic JSON redacted]"],
            },
        },
    )
    export = parse_trajectory_export(blob)
    assert export.errors == []
    assert export.output == "Deployed hello-app; 3/3 pods Ready."


def test_parse_trajectory_export_placeholder_does_not_clobber_an_earlier_turn() -> None:
    """A redacted last turn leaves the previous turn's real assistantTexts standing."""
    blob = _events(
        {"type": "model.completed", "data": {"assistantTexts": ["Applying the manifest now."]}},
        {
            "type": "model.completed",
            "data": {"assistantTexts": ["[Oversized diagnostic JSON redacted]"]},
        },
    )
    assert parse_trajectory_export(blob).output == "Applying the manifest now."


def test_parse_trajectory_export_placeholder_assistant_message_is_not_recovered() -> None:
    """The fallback skips redacted assistant.messages instead of returning them.

    When every source is a placeholder the parser returns ``""``, which is the
    shape the scoring layer's missing-answer rule keys on — better an empty
    output than a judge grading the sanitizer's stand-in.
    """
    blob = _events(
        {
            "type": "assistant.message",
            "data": {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "[Malformed diagnostic JSON redacted]"}],
                }
            },
        },
        {
            "type": "model.completed",
            "data": {"assistantTexts": ["[Malformed diagnostic JSON redacted]"]},
        },
    )
    assert parse_trajectory_export(blob).output == ""


def test_parse_trajectory_export_answer_quoting_a_placeholder_is_kept() -> None:
    """Only a whole-string placeholder is dropped; an answer about one survives."""
    text = "Retried after seeing '[Malformed diagnostic JSON redacted]' in the log; done."
    blob = _events({"type": "model.completed", "data": {"assistantTexts": [text]}})
    assert parse_trajectory_export(blob).output == text


def test_parse_trajectory_export_surfaces_decode_errors() -> None:
    blob = "{not json}\n" + json.dumps(_tool_call("1", "x", {})) + "\n"
    export = parse_trajectory_export(blob)
    assert any("parse error" in m for m in export.errors)
    assert len(export.trajectory) == 1


def test_parse_trajectory_export_drops_unpaired_result_and_surfaces_error() -> None:
    """Orphan tool.result is dropped from trajectory, recorded on errors.

    Mirrors the API agent's ``_fold_with_extraction_errors`` policy and the
    Gemini ``parse_stream_json`` policy so every agent feeds the metrics seam
    one shape — only real ToolCalls the model issued ride on
    ``AgentResult.trajectory``; orphans are diagnostics, not trajectory entries.
    """
    blob = _events(_tool_result("ghost", "?"))
    export = parse_trajectory_export(blob)
    # Orphan must NOT appear in the canonical trajectory.
    assert export.trajectory == []
    # ...but MUST be surfaced on errors so the run is never silent-empty.
    assert any("without matching call" in m for m in export.errors)
    assert any("ghost" in m for m in export.errors)


def test_strip_ansi_removes_color_codes() -> None:
    assert _strip_ansi("\x1b[31mhello\x1b[0m") == "hello"


def test_oc_model_id_normalizes_provider_alias() -> None:
    assert _oc_model_id(AgentConfig(model="gemini-2.5-pro", provider="gemini")) == (
        "google/gemini-2.5-pro"
    )


def test_oc_model_id_preserves_full_id() -> None:
    assert _oc_model_id(AgentConfig(model="anthropic/claude-opus-4-7")) == (
        "anthropic/claude-opus-4-7"
    )


def test_oc_model_id_normalizes_full_id_provider_segment() -> None:
    # A full id whose wire is an alias is normalized through the contract.
    assert _oc_model_id(AgentConfig(model="gemini/gemini-2.5-pro")) == ("google/gemini-2.5-pro")


def test_oc_model_id_passes_through_unknown_full_id_wire() -> None:
    # An unrecognized wire is left for oc to validate, not rejected here.
    assert _oc_model_id(AgentConfig(model="mystery/some-model")) == "mystery/some-model"


def test_oc_model_id_returns_empty_when_no_model() -> None:
    assert _oc_model_id(AgentConfig()) == ""


def test_oc_model_id_defaults_to_google() -> None:
    assert _oc_model_id(AgentConfig(model="gemini-2.5-pro")) == "google/gemini-2.5-pro"


def test_build_local_command_quotes_inputs_and_passes_model_flag() -> None:
    cfg = AgentConfig(model="gemini-2.5-pro", provider="gemini")
    cmd = _build_local_command(cfg, "hi 'world'", "main", "/usr/local/bin/oc")
    # Prompt single-quote must be escaped, not break the shell line.
    assert "hi '\\''world'\\''" in cmd or "hi 'world'" not in cmd
    assert "NVM_DIR" in cmd  # nvm sourced for the node runtime
    # Per-run model override, not the global `oc models set` (no shared config write).
    assert "--model google/gemini-2.5-pro" in cmd
    assert "models set" not in cmd
    assert "agent --local" in cmd
    assert "--agent main" in cmd


def test_build_local_command_omits_model_flag_when_no_model_configured() -> None:
    cmd = _build_local_command(AgentConfig(), "prompt", "main", "/usr/local/bin/oc")
    assert "--model" not in cmd
    assert "models set" not in cmd


def test_build_local_command_forwards_extra_flags() -> None:
    cfg = AgentConfig(extra_flags=("--flag1", "--opt=val", "--quoted=hello world"))
    cmd = _build_local_command(cfg, "prompt", "main", "/usr/local/bin/oc")
    assert "--flag1" in cmd
    assert "--opt=val" in cmd
    assert "'--quoted=hello world'" in cmd


def test_pick_session_key_handles_top_level_list() -> None:
    payload = json.dumps([{"key": "agent:operator:abc", "model": "x"}])
    assert _pick_session_key(payload) == "agent:operator:abc"


def test_pick_session_key_handles_wrapper_dict() -> None:
    payload = json.dumps({"sessions": [{"key": "k1"}, {"key": "k2"}]})
    assert _pick_session_key(payload) == "k1"


def test_pick_session_key_returns_none_on_empty_or_invalid() -> None:
    assert _pick_session_key("") is None
    assert _pick_session_key("{}") is None
    assert _pick_session_key(json.dumps({"sessions": []})) is None
    assert _pick_session_key(json.dumps([{"no_key": "x"}])) is None


def test_openclaw_agent_registered_under_canonical_key() -> None:
    assert AGENTS.get("openclaw") is OpenClawAgent


def test_ensure_node_on_path_picks_numerically_newest_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """v18 must beat v8 even though "v18" < "v8" as strings."""
    for version in ("v8.9.0", "v18.17.0", "v9.11.2"):
        (tmp_path / "versions" / "node" / version / "bin").mkdir(parents=True)
    monkeypatch.setenv("NVM_DIR", str(tmp_path))
    monkeypatch.setattr(oc_mod.shutil, "which", lambda _cmd: None)

    merged = oc_mod._ensure_node_on_path({})

    first_path_entry = merged["PATH"].split(os.pathsep)[0]
    assert first_path_entry == str(tmp_path / "versions" / "node" / "v18.17.0" / "bin")


def _make_subprocess_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _install_oc_run(
    monkeypatch: pytest.MonkeyPatch,
    fake_bash: Callable[..., Any],
    fake_core_run: Callable[..., Any] | None = None,
) -> None:
    """Install a fake ``core.subprocess.run`` on the agent module.

    The agent turn arrives as ``["/bin/bash", "-c", <command>]`` and is routed
    to ``fake_bash(<command>, **kwargs)`` so fakes keep asserting on the bash
    command string; any other argv is the direct ``oc`` extraction call and
    goes to ``fake_core_run`` (defaulting to an empty ``oc sessions``).
    """
    core = fake_core_run or (lambda argv, **kwargs: _make_subprocess_result(json.dumps([]), "", 0))

    def dispatch(argv, **kwargs):
        if argv[0] == "/bin/bash":
            return fake_bash(argv[2], **kwargs)
        return core(argv, **kwargs)

    monkeypatch.setattr(oc_mod, "run", dispatch)


def _bundle_writer(events_jsonl: str) -> Callable[..., Any]:
    """Build a fake core-subprocess.run that writes an export bundle's events.jsonl.

    ``oc sessions`` returns one session row; ``oc sessions export-trajectory``
    writes ``events.jsonl`` (the real trajectory filename) into the bundle dir
    under the ``--workspace`` it was handed.
    """
    sessions_payload = json.dumps(
        [{"key": "agent:operator:test", "model": "google/gemini-2.5-pro"}]
    )

    def fake_core_run(argv, **kwargs):
        if argv[1] == "sessions" and "export-trajectory" not in argv:
            return _make_subprocess_result(stdout=sessions_payload, returncode=0)
        if "export-trajectory" in argv:
            ws = Path(argv[argv.index("--workspace") + 1])
            export_root = ws / ".openclaw" / "trajectory-exports" / "openclaw-trajectory-x"
            export_root.mkdir(parents=True, exist_ok=True)
            (export_root / "events.jsonl").write_text(events_jsonl, encoding="utf-8")
            return _make_subprocess_result(stdout="exported", returncode=0)
        raise AssertionError(f"unexpected argv {argv}")

    return fake_core_run


def test_execute_happy_path_emits_canonical_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """oc agent succeeds, oc sessions yields one row, export-trajectory parses cleanly.

    The final answer comes from the bundle (``model.completed.assistantTexts``),
    not the noisy bash stdout.
    """

    def fake_bash(cmd, **kwargs):
        return _make_subprocess_result(stdout="OK\n", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _bundle_writer(SAMPLE_EVENTS))

    agent = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=30.0))
    result = agent.run("audit pods in default")
    assert result.errors == []
    assert result.terminal_reason == "completed"
    assert len(result.trajectory) == 2
    assert result.trajectory[0]["name"] == "kubectl_get_pods"
    assert result.tokens == {"input": 5, "output": 10, "total": 15}
    assert result.output == "All pods healthy."


def test_execute_prefers_bundle_output_over_noisy_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The agent's final answer (events.jsonl assistantTexts) must win over
    `oc --log-level debug` noise — otherwise the judge grades debug spew.
    """
    noisy_stdout = "[DEBUG] starting oc...\n[INFO] sessionFile=/tmp/.openclaw/...\n[DEBUG] turn 1\n"

    def fake_bash(cmd, **kwargs):
        return _make_subprocess_result(stdout=noisy_stdout, returncode=0)

    clean_answer = "All pods in `default` are Running."
    events = _events(
        _tool_call("1", "exec", {"command": "kubectl get pods"}),
        _tool_result("1", "pod-a Running"),
        {
            "type": "model.completed",
            "data": {"usage": {"input": 1, "output": 1}, "assistantTexts": [clean_answer]},
        },
    )
    _install_oc_run(monkeypatch, fake_bash, _bundle_writer(events))

    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("audit pods")
    assert result.output == clean_answer
    assert "[DEBUG]" not in result.output


def test_execute_falls_back_to_stdout_when_bundle_has_no_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No assistantTexts and no assistant.message in events → use stripped stdout."""
    events = _events(
        _tool_call("1", "exec", {"command": "ls"}),
        _tool_result("1", "file"),
        {"type": "model.completed", "data": {"usage": {"input": 1, "output": 1}}},
    )
    _install_oc_run(
        monkeypatch,
        lambda *a, **k: _make_subprocess_result(stdout="bare stdout answer", returncode=0),
        _bundle_writer(events),
    )

    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert result.output == "bare stdout answer"


def test_oc_version_probe_parses_a_prerelease_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ``__wrapped__`` bypasses the @cache so the fake run is actually hit.
    monkeypatch.setattr(
        oc_mod, "run", lambda argv, **kw: _make_subprocess_result("2026.9.1-beta.1\n", "", 0)
    )
    assert oc_mod._host_oc_version.__wrapped__("oc") == "2026.9.1-beta.1"
    assert oc_mod._image_oc_version.__wrapped__("img") == "2026.9.1-beta.1"


def _raise_oserror(argv, **kw):
    raise OSError("docker not installed")


def test_oc_version_probe_is_inconclusive_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wrapper without a --version surface, or a docker failure, must yield
    # None (probe inconclusive) rather than raising — refusing to run on a
    # failed probe would be worse than the risk it guards.
    monkeypatch.setattr(
        oc_mod, "run", lambda argv, **kw: _make_subprocess_result("no version here", "", 1)
    )
    assert oc_mod._host_oc_version.__wrapped__("oc-a") is None
    monkeypatch.setattr(oc_mod, "run", _raise_oserror)
    assert oc_mod._image_oc_version.__wrapped__("img-a") is None


def test_oc_version_skew_names_both_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oc_mod, "_host_oc_version", lambda b: "2026.8.2")
    monkeypatch.setattr(oc_mod, "_image_oc_version", lambda i: "2026.9.1-beta.1")
    skew = oc_mod._oc_version_skew("oc", "img")
    assert skew is not None
    assert "2026.8.2" in skew
    assert "2026.9.1-beta.1" in skew


def test_oc_version_skew_silent_on_match_or_inconclusive_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(oc_mod, "_host_oc_version", lambda b: "2026.9.1-beta.1")
    monkeypatch.setattr(oc_mod, "_image_oc_version", lambda i: "2026.9.1-beta.1")
    assert oc_mod._oc_version_skew("oc", "img") is None
    monkeypatch.setattr(oc_mod, "_host_oc_version", lambda b: None)
    assert oc_mod._oc_version_skew("oc", "img") is None


def test_execute_fails_fast_on_sandboxed_version_skew(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A skewed host/image oc pair errors before the agent turn ever launches.

    The live failure this guards: the run itself succeeded, but the post-run
    export refused the container-written session store ("written by version
    2026.9.1-beta.1, but this command is running 2026.8.2"), burning a full
    cluster spin-up to produce an empty trajectory stamped success.
    """
    monkeypatch.setattr(oc_mod, "_oc_version_skew", lambda b, i: "oc version skew: boom")

    def never_run(argv, **kw):
        raise AssertionError("the agent turn must not launch on a skewed pair")

    monkeypatch.setattr(oc_mod, "run", never_run)
    agent = OpenClawAgent(
        AgentConfig(target=str(tmp_path / "oc"), sandbox=SandboxSpec(image="img"))
    )
    result = agent._execute("p")
    assert result.has_errors()
    assert any("version skew" in e for e in result.errors)
    assert result.trajectory == []


def test_execute_unsandboxed_skips_the_version_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def never_probe(b, i):
        raise AssertionError("no sandbox image, so nothing to compare against")

    monkeypatch.setattr(oc_mod, "_oc_version_skew", never_probe)
    _install_oc_run(
        monkeypatch,
        lambda *a, **k: _make_subprocess_result("OK\n", "", 0),
        _bundle_writer(SAMPLE_EVENTS),
    )
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert result.errors == []


def test_execute_keeps_the_export_bundle_in_a_harness_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The raw events.jsonl survives the run when the harness owns the workspace.

    Exported into the run workdir (not a throwaway temp dir), the bundle's
    lifetime follows the workspace's, and the harness's existing workspace
    diff carries it into the run's generated_files. The first observed
    oc-side redaction could not be root-caused because the bundle had died
    with the temp dir this replaces.
    """
    _install_oc_run(
        monkeypatch,
        lambda *a, **k: _make_subprocess_result("OK\n", "", 0),
        _bundle_writer(SAMPLE_EVENTS),
    )
    workspace = tmp_path / "run-ws"
    workspace.mkdir()

    agent = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=30.0))
    result = agent._execute("audit pods", workspace_path=workspace)

    assert result.output == "All pods healthy."
    bundles = list((workspace / ".openclaw" / "trajectory-exports").rglob("events.jsonl"))
    assert len(bundles) == 1
    assert json.loads(bundles[0].read_text().splitlines()[0])["type"] == "tool.call"


def test_execute_records_when_sessions_returns_no_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_core_run(argv, **kwargs):
        # oc sessions returns empty.
        return _make_subprocess_result(stdout=json.dumps([]), returncode=0)

    _install_oc_run(
        monkeypatch, lambda *a, **k: _make_subprocess_result("ok", "", 0), fake_core_run
    )

    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert result.has_errors()
    assert any("no session key" in e for e in result.errors)


def test_execute_records_export_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_core_run(argv, **kwargs):
        if "export-trajectory" in argv:
            raise SubprocessError(argv, returncode=1, stdout="", stderr="bad")
        return _make_subprocess_result(stdout=json.dumps([{"key": "k1"}]), returncode=0)

    _install_oc_run(
        monkeypatch, lambda *a, **k: _make_subprocess_result("ok", "", 0), fake_core_run
    )
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert result.has_errors()
    assert any("export-trajectory failed" in e for e in result.errors)


def test_execute_records_bash_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_bash(cmd, **kwargs):
        # core.subprocess.run wraps TimeoutExpired in SubprocessError.
        raise SubprocessError(
            ["/bin/bash", "-c", cmd], returncode=-1, stdout="", stderr="", timed_out=True
        )

    _install_oc_run(monkeypatch, fake_bash)
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=5.0)).run("p")
    assert result.has_errors()
    assert "timed out" in result.errors[0]
    assert result.trajectory == []
    assert result.terminal_reason == "timeout"


def test_execute_recovers_telemetry_from_a_timed_out_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A killed agent turn still exported its session; the row keeps that telemetry.

    The budget ends the turn, not the session on disk, and export-trajectory is
    a separate subprocess. Nulling the trajectory here would blank the counters
    on exactly the rows where "how far did it get" is the question.
    """

    def fake_bash(cmd, **kwargs):
        raise SubprocessError(
            ["/bin/bash", "-c", cmd], returncode=-1, stdout="", stderr="", timed_out=True
        )

    _install_oc_run(monkeypatch, fake_bash, _bundle_writer(SAMPLE_EVENTS))
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=5.0)).run("p")
    assert result.terminal_reason == "timeout"
    assert any("timed out" in e for e in result.errors)
    # The partial run's work survives rather than being discarded with the kill.
    assert len(result.trajectory) == 2
    assert result.tokens == {"input": 5, "output": 10, "total": 15}
    assert result.output == "All pods healthy."


def test_execute_non_timeout_subprocess_error_is_not_labelled_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``terminal_reason`` follows ``exc.timed_out``, not the exception type."""

    def fake_bash(cmd, **kwargs):
        raise SubprocessError(
            ["/bin/bash", "-c", cmd], returncode=-1, stdout="", stderr="", timed_out=False
        )

    _install_oc_run(monkeypatch, fake_bash)
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=5.0)).run("p")
    assert result.terminal_reason == "error"
    assert not any("timed out" in e for e in result.errors)


def test_execute_passes_timeout_to_bash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        captured.update(kwargs)
        return _make_subprocess_result("ok", "", 0)

    _install_oc_run(monkeypatch, fake_bash)
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=12.5)).run("p")
    assert captured["timeout"] == 12.5


class _FakeClock:
    """Monotonic clock that only moves when a fake explicitly advances it."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


def _install_fake_clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(oc_mod, "time", SimpleNamespace(monotonic=clock.monotonic))
    return clock


def test_latency_excludes_the_trajectory_export(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reported latency is the agent turn, not the turn plus our export.

    Exporting the trajectory afterwards is harness work; billing it to the
    agent makes openclaw look slower than a harness that has no export step.
    """
    clock = _install_fake_clock(monkeypatch)

    def fake_bash(cmd, **kwargs):
        clock.now += 5.0
        return _make_subprocess_result("ok", "", 0)

    def fake_core_run(argv, **kwargs):
        clock.now += 100.0
        return _make_subprocess_result(json.dumps([]), "", 0)

    _install_oc_run(monkeypatch, fake_bash, fake_core_run)
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert result.latency == 5.0


def test_timeout_result_carries_the_elapsed_agent_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = _install_fake_clock(monkeypatch)

    def fake_bash(cmd, **kwargs):
        clock.now += 7.0
        raise SubprocessError(
            ["/bin/bash", "-c", cmd], returncode=-1, stdout="", stderr="", timed_out=True
        )

    _install_oc_run(monkeypatch, fake_bash)
    result = OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), timeout_sec=7.0)).run("p")
    assert result.latency == 7.0
    assert result.terminal_reason == "timeout"


# Tests for the now-deleted legacy surface — fail-fast if SSH transport returns.


def test_legacy_ssh_runner_is_gone() -> None:
    assert not hasattr(oc_mod, "run_openclaw_agent")


def test_legacy_local_runner_is_gone() -> None:
    assert not hasattr(oc_mod, "run_openclaw_agent_local")


# ---------------------------------------------------------------------------
# Capability negotiation: OpenClaw wires MCP + skills via oc's native channels
# ---------------------------------------------------------------------------


def test_openclaw_satisfies_mcp_skills_and_rules_protocols() -> None:
    """OpenClaw declares MCP, Skills and Rules: it writes ``mcp.servers`` into an
    isolated ``OPENCLAW_CONFIG_PATH``, materializes managed skills under
    ``<OPENCLAW_STATE_DIR>/skills``, and prepends the operator brief to the
    prompt."""
    agent = OpenClawAgent(AgentConfig())
    assert isinstance(agent, SupportsRules)
    assert isinstance(agent, SupportsMcp)
    assert isinstance(agent, SupportsSkills)


def test_openclaw_agent_mirrors_capability_bindings_onto_mixin_attributes() -> None:
    """The structural-Protocol attributes track the granted bindings."""
    binding = McpBinding(name="gke", command=("gke-mcp",), tools=("t",))
    skills = SkillBinding(paths=("/some/skills",))
    caps = AllCapabilities(
        mcp_servers=(binding,),
        skills=skills,
        rules=AgentRules(text="be a sre"),
    )
    agent = OpenClawAgent(AgentConfig(capabilities=caps))
    assert agent.mcp_servers == (binding,)
    assert agent.skills == skills
    assert agent.rules == AgentRules(text="be a sre")


def test_openclaw_agent_mirrors_rules_binding_onto_mixin_attribute() -> None:
    caps = AllCapabilities(rules=AgentRules(text="be precise"))
    agent = OpenClawAgent(AgentConfig(capabilities=caps))
    assert agent.rules == AgentRules(text="be precise")


# ---------------------------------------------------------------------------
# Rules delivery: the bound text actually reaches the spawned `oc` command.
# ---------------------------------------------------------------------------


def test_prepend_rules_passes_prompt_through_when_rules_empty() -> None:
    from devops_bench.agents.cli.openclaw.agent import _prepend_rules

    assert _prepend_rules("", "do the thing") == "do the thing"
    assert _prepend_rules("   \n  ", "do the thing") == "do the thing"


def test_prepend_rules_separates_brief_from_prompt_with_blank_line() -> None:
    from devops_bench.agents.cli.openclaw.agent import _prepend_rules

    assert _prepend_rules("be careful", "audit pods") == "be careful\n\naudit pods"


def test_execute_prepends_bound_rules_to_oc_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The rules text must land inside the bash command string the agent
    spawns — specifically inside the ``-m '<prompt>'`` segment that ``oc
    agent`` reads. We capture the command and assert both the original prompt
    and the rules text are present, with the rules ahead of the prompt."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        captured["cmd"] = cmd
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash)

    caps = AllCapabilities(rules=AgentRules(text="you are a precise SRE"))
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), capabilities=caps)).run(
        "audit pods in default"
    )

    cmd = captured["cmd"]
    assert "you are a precise SRE" in cmd, "rules text must reach the spawned oc command"
    assert "audit pods in default" in cmd, "task prompt must still reach oc"
    # The rules brief must precede the task prompt — order matters because the
    # model reads it as the leading context.
    assert cmd.index("you are a precise SRE") < cmd.index("audit pods in default")


def test_execute_does_not_prepend_rules_when_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With default (empty) rules, the prompt reaches oc unchanged — no
    accidental blank-line prefix that would shift the model's attention."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        captured["cmd"] = cmd
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash)

    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("just the prompt")
    cmd = captured["cmd"]
    # The agent shlex-quotes the prompt; "just the prompt" appears verbatim
    # inside single quotes in the `-m` segment — and no extra blank-line
    # prefix surrounds it.
    assert "-m 'just the prompt'" in cmd


# ---------------------------------------------------------------------------
# MCP server wiring: mcp.servers reach an isolated OPENCLAW_CONFIG_PATH.
# ---------------------------------------------------------------------------


def test_build_openclaw_config_wraps_servers_under_mcp() -> None:
    """A launchable binding renders under the ``mcp.servers`` config path."""
    cfg = _build_openclaw_config(AgentConfig(), (McpBinding(name="gke", command=("gke-mcp",)),))
    assert cfg == {
        "mcp": {"servers": {"gke": {"command": "gke-mcp"}}},
        "tools": {"codeMode": False, "deny": ["sessions_spawn", "sessions_yield"]},
        "memory": {"search": {"enabled": False}},
    }


def test_build_openclaw_config_merges_mcp_and_model_override() -> None:
    """MCP servers and a model-catalog entry coexist (disjoint key spaces)."""
    cfg = _build_openclaw_config(
        AgentConfig(model="gemini-3.5-flash", provider="google"),
        (McpBinding(name="gke", command=("gke-mcp",)),),
    )
    assert cfg["mcp"] == {"servers": {"gke": {"command": "gke-mcp"}}}
    assert cfg["models"]["providers"]["google"]["models"] == [
        {"id": "gemini-3.5-flash", "name": "gemini-3.5-flash"}
    ]
    assert cfg["agents"]["defaults"]["models"] == {"google/gemini-3.5-flash": {}}


# ---------------------------------------------------------------------------
# Model catalog override: models oc doesn't ship by default get registered in
# the per-run isolated config, for both google-genai and google-vertex.
# ---------------------------------------------------------------------------


def test_model_override_empty_for_catalog_known_model() -> None:
    """A model already in oc's catalog needs no override."""
    assert _build_model_override(AgentConfig(model="gemini-3.1-pro-preview")) == {}


def test_model_override_empty_when_no_model() -> None:
    assert _build_model_override(AgentConfig()) == {}


def test_model_override_genai_pins_generative_ai_transport() -> None:
    """google-genai: the entry pins ``api: google-generative-ai`` so oc routes it
    through the google-genai transport (a per-run provider entry replaces oc's
    built-in one, so the transport must be carried) and needs no ``baseUrl``.
    Allowlists ``google/<model>``."""
    override = _build_model_override(AgentConfig(model="gemini-3.5-flash", provider="google"))
    google = override["models"]["providers"]["google"]
    assert google["api"] == "google-generative-ai"
    assert "baseUrl" not in google
    assert google["models"] == [{"id": "gemini-3.5-flash", "name": "gemini-3.5-flash"}]
    assert override["agents"]["defaults"]["models"] == {"google/gemini-3.5-flash": {}}


def test_model_override_gemini_alias_normalizes_to_google() -> None:
    """``provider=gemini`` resolves to the ``google`` provider (id alias)."""
    override = _build_model_override(AgentConfig(model="gemini-3.5-flash", provider="gemini"))
    assert "google" in override["models"]["providers"]
    assert override["agents"]["defaults"]["models"] == {"google/gemini-3.5-flash": {}}


def test_model_override_vertex_pins_transport_and_allowlists() -> None:
    """google-vertex: the entry pins ``api``/``baseUrl`` so oc uses the vertex
    transport (not the OpenAI fallback), and allowlists ``google-vertex/<model>``."""
    override = _build_model_override(
        AgentConfig(model="gemini-3.5-flash", provider="google-vertex")
    )
    vertex = override["models"]["providers"]["google-vertex"]
    assert vertex["api"] == "google-vertex"
    assert vertex["baseUrl"] == "https://{location}-aiplatform.googleapis.com"
    assert vertex["models"] == [{"id": "gemini-3.5-flash", "name": "gemini-3.5-flash"}]
    assert override["agents"]["defaults"]["models"] == {"google-vertex/gemini-3.5-flash": {}}


def test_model_override_vertex_needs_no_api_key() -> None:
    """The override is independent of ``api_key`` — a keyless ADC vertex run still
    gets its catalog entry (auth is the metadata-server ADC, set outside)."""
    override = _build_model_override(
        AgentConfig(model="gemini-3.5-flash", provider="google-vertex", api_key=None)
    )
    assert override["agents"]["defaults"]["models"] == {"google-vertex/gemini-3.5-flash": {}}
    assert _build_env(AgentConfig(model="gemini-3.5-flash", provider="google-vertex")) == {}


def test_build_env_threads_api_key_by_provider() -> None:
    """``config.api_key`` lands on the provider-specific env var(s)."""
    google = _build_env(AgentConfig(api_key="k", provider="google"))
    assert google["GEMINI_API_KEY"] == "k" and google["GOOGLE_API_KEY"] == "k"
    anthropic = _build_env(AgentConfig(api_key="k", provider="anthropic"))
    assert anthropic == {"ANTHROPIC_API_KEY": "k"}
    assert _build_env(AgentConfig()) == {}


def test_build_env_routes_vertex_key_to_cloud_api_key() -> None:
    """A ``google-vertex`` key reaches ``GOOGLE_CLOUD_API_KEY`` (the vertex
    transport's var), not ``GEMINI_API_KEY`` (the google-genai one)."""
    vertex = _build_env(AgentConfig(api_key="marker", provider="google-vertex"))
    assert vertex == {"GOOGLE_CLOUD_API_KEY": "marker"}


def test_build_env_unknown_provider_raises() -> None:
    """An unknown provider fails loud instead of silently writing a Gemini key."""
    with pytest.raises(ConfigError):
        _build_env(AgentConfig(api_key="k", provider="mystery"))


def test_build_env_unknown_provider_raises_even_when_keyless() -> None:
    """Validation is unconditional — a typoed provider fails loud on a keyless
    (Vertex/ADC) run too, not only when a key is set."""
    with pytest.raises(ConfigError):
        _build_env(AgentConfig(provider="google-vertyx"))


def test_model_override_raises_for_unpinned_transport() -> None:
    """A catalog-override model whose provider has no pinned transport fails loud
    rather than shipping a transport-less entry (which would 401 via the OpenAI
    fallback). A full-id with an unknown wire reaches this path."""
    with pytest.raises(ConfigError):
        _build_model_override(AgentConfig(model="mystery/gemini-3.5-flash"))


def _empty_sessions_run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
    """Core-subprocess.run stub: ``oc sessions`` returns no rows."""
    return _make_subprocess_result(stdout=json.dumps([]), returncode=0)


def test_execute_writes_mcp_servers_into_isolated_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A command-bearing MCP binding lands in ``<cwd>/openclaw.json`` and
    ``OPENCLAW_CONFIG_PATH`` is pointed at it (read inside the fake to beat
    temp-dir cleanup)."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        env = kwargs.get("extra_env") or {}
        cfg_path = env.get("OPENCLAW_CONFIG_PATH")
        captured["cfg_path"] = cfg_path
        captured["config"] = (
            json.loads(Path(cfg_path).read_text()) if cfg_path and Path(cfg_path).exists() else None
        )
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    caps = AllCapabilities(
        mcp_servers=(McpBinding(name="gke", command=("gke-mcp",), tools=("mcp_gke_x",)),),
    )
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), capabilities=caps)).run("p")
    assert captured["cfg_path"], "OPENCLAW_CONFIG_PATH must be set when MCP is bound"
    assert captured["config"] == {
        "mcp": {"servers": {"gke": {"command": "gke-mcp"}}},
        "tools": {"codeMode": False, "deny": ["sessions_spawn", "sessions_yield"]},
        "memory": {"search": {"enabled": False}},
    }


def test_execute_writes_model_override_config_without_mcp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model absent from oc's catalog gets an isolated config + OPENCLAW_CONFIG_PATH
    even with no MCP server — and works keyless (vertex/ADC)."""
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        env = kwargs.get("extra_env") or {}
        cfg_path = env.get("OPENCLAW_CONFIG_PATH")
        captured["cfg_path"] = cfg_path
        captured["config"] = (
            json.loads(Path(cfg_path).read_text()) if cfg_path and Path(cfg_path).exists() else None
        )
        # ADC path: no API key threaded into the subprocess env.
        captured["has_key"] = any(
            k in env for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_API_KEY")
        )
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    OpenClawAgent(
        AgentConfig(
            target=str(tmp_path / "oc"),
            model="gemini-3.5-flash",
            provider="google-vertex",
        )
    ).run("p")
    assert captured["cfg_path"], "OPENCLAW_CONFIG_PATH must be set for a catalog override"
    vertex = captured["config"]["models"]["providers"]["google-vertex"]
    assert vertex["api"] == "google-vertex"
    assert captured["config"]["agents"]["defaults"]["models"] == {
        "google-vertex/gemini-3.5-flash": {}
    }
    assert captured["has_key"] is False


def test_execute_isolates_state_dir_and_drops_global_session_wipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``OPENCLAW_STATE_DIR`` points under the per-run cwd and the old global
    ``rm -rf ~/.openclaw/.../sessions`` wipe is gone (state is fresh per run)."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        env = kwargs.get("extra_env") or {}
        captured["state_dir"] = env.get("OPENCLAW_STATE_DIR")
        captured["cwd"] = kwargs.get("cwd")
        captured["cmd"] = cmd
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert captured["state_dir"] == os.path.join(captured["cwd"], "state")
    assert "rm -rf" not in captured["cmd"]


def test_execute_threads_api_key_into_subprocess_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model API key reaches the spawned ``oc`` process env."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        captured["env"] = kwargs.get("extra_env") or {}
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    cfg = AgentConfig(target=str(tmp_path / "oc"), api_key="secret", provider="google")
    OpenClawAgent(cfg).run("p")
    assert captured["env"]["GEMINI_API_KEY"] == "secret"
    assert captured["env"]["GOOGLE_API_KEY"] == "secret"


def test_execute_materializes_skills_into_state_skills_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Bound skill paths are copied to ``<cwd>/state/skills/<name>/SKILL.md``."""
    src = tmp_path / "skills" / "my-skill"
    src.mkdir(parents=True)
    skill_text = "---\nname: my-skill\ndescription: do things\n---\nbody\n"
    (src / "SKILL.md").write_text(skill_text)

    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        cwd = kwargs.get("cwd")
        skill_path = os.path.join(cwd, "state", "skills", "my-skill", "SKILL.md")
        captured["exists"] = os.path.exists(skill_path)
        captured["text"] = Path(skill_path).read_text() if captured["exists"] else None
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    caps = AllCapabilities(skills=SkillBinding(paths=(str(tmp_path / "skills"),)))
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), capabilities=caps)).run("p")
    assert captured["exists"], "skill must be materialized before subprocess"
    assert captured["text"] == skill_text


def test_execute_warns_and_skips_missing_skill_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-existent skill path is skipped (no crash, empty skills root)."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        cwd = kwargs.get("cwd")
        skills_root = os.path.join(cwd, "state", "skills")
        captured["entries"] = sorted(os.listdir(skills_root)) if os.path.isdir(skills_root) else []
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    caps = AllCapabilities(skills=SkillBinding(paths=("/no/such/skills/dir",)))
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"), capabilities=caps)).run("p")
    assert captured["entries"] == []


def test_execute_cleans_up_temp_working_dir_after_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The per-run temp dir (state, config, skills) is removed after _execute."""
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        captured["cwd"] = kwargs.get("cwd")
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    OpenClawAgent(AgentConfig(target=str(tmp_path / "oc"))).run("p")
    assert captured["cwd"] is not None
    assert not os.path.exists(captured["cwd"])


def test_build_openclaw_config_carries_only_code_mode_without_launchable_server_or_override() -> (
    None
):
    """No MCP binding and a catalog-known model → only ``tools.codeMode``/``deny``."""
    expected = {
        "tools": {"codeMode": False, "deny": ["sessions_spawn", "sessions_yield"]},
        "memory": {"search": {"enabled": False}},
    }
    assert _build_openclaw_config(AgentConfig(), ()) == expected
    assert (
        _build_openclaw_config(
            AgentConfig(model="gemini-3.1-pro-preview"),
            (McpBinding(name="b", command=(), tools=("t",)),),
        )
        == expected
    )


def test_build_openclaw_config_defaults_code_mode_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCH_OPENCLAW_CODE_MODE", raising=False)
    payload = _build_openclaw_config(AgentConfig(), ())
    assert payload["tools"]["codeMode"] is False


def test_build_openclaw_config_honors_code_mode_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCH_OPENCLAW_CODE_MODE", "1")
    payload = _build_openclaw_config(AgentConfig(), ())
    assert payload["tools"]["codeMode"] is True


def test_build_openclaw_config_denies_delegation_tools() -> None:
    """sessions_spawn/sessions_yield are denied regardless of provider: the bench
    always runs embedded (``oc agent --local``), so a delegated subagent's
    completion never arrives and the run would end having done no work."""
    cfg = _build_openclaw_config(AgentConfig(), ())
    assert set(cfg["tools"]["deny"]) == {"sessions_spawn", "sessions_yield"}


def test_build_openclaw_config_deny_does_not_clobber_code_mode() -> None:
    """The deny list coexists with ``tools.codeMode``; neither overwrites the other."""
    cfg = _build_openclaw_config(AgentConfig(), ())
    assert cfg["tools"] == {
        "codeMode": False,
        "deny": ["sessions_spawn", "sessions_yield"],
    }


def test_build_openclaw_config_pins_openai_to_builtin_agent_runtime() -> None:
    """openai provider gets an explicit agentRuntime so oc skips the codex route."""
    cfg = _build_openclaw_config(AgentConfig(model="gpt-5.6-sol", provider="openai"), ())
    assert cfg["models"]["providers"]["openai"]["agentRuntime"] == {"id": "openclaw"}


def test_build_openclaw_config_omits_agent_runtime_for_non_openai_provider() -> None:
    """A gemini/google-vertex config gets no agentRuntime override; it's inert today."""
    cfg = _build_openclaw_config(AgentConfig(model="gemini-2.5-pro", provider="google-vertex"), ())
    assert "models" not in cfg or "openai" not in cfg["models"].get("providers", {})


def test_build_openclaw_config_agent_runtime_does_not_clobber_model_override() -> None:
    """The openai agentRuntime merge coexists with a catalog override for another provider."""
    cfg = _build_openclaw_config(AgentConfig(model="gemini-3.5-flash", provider="google"), ())
    assert cfg["models"]["providers"]["google"]["models"] == [
        {"id": "gemini-3.5-flash", "name": "gemini-3.5-flash"}
    ]
    assert "openai" not in cfg["models"]["providers"]


def test_build_openclaw_config_pins_openai_runtime_for_full_model_id() -> None:
    """A full ``openai/...`` id selects the openai runtime with no explicit
    provider set. Reading ``config.provider`` alone resolves to the default
    provider, drops the pin, and lets oc route the run to the absent codex
    runtime."""
    cfg = _build_openclaw_config(AgentConfig(model="openai/gpt-5.6-sol"), ())
    assert cfg["models"]["providers"]["openai"]["agentRuntime"] == {"id": "openclaw"}


# ---------------------------------------------------------------------------
# Model catalog override: models oc doesn't ship by default get registered in
# the per-run isolated config, for both google-genai and google-vertex.
# ---------------------------------------------------------------------------


def test_build_openclaw_config_disables_memory_search() -> None:
    """openclaw 2026.8.x enables memory search by default, which both calls an
    OpenAI embeddings endpoint the run never selected and lets one run recall a
    previous run of the same task. A benchmark run must be stateless."""
    payload = _build_openclaw_config(AgentConfig(), ())
    assert payload["memory"] == {"search": {"enabled": False}}


def test_execute_writes_code_mode_config_when_no_launchable_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No command-bearing MCP binding and a catalog-known model → the isolated
    config still carries ``tools.codeMode``, so the env var is still set."""
    captured: dict = {}

    def fake_bash(cmd: str, **kwargs: Any) -> SimpleNamespace:
        env = kwargs.get("extra_env") or {}
        cfg_path = env.get("OPENCLAW_CONFIG_PATH")
        captured["cfg_path"] = cfg_path
        captured["config"] = (
            json.loads(Path(cfg_path).read_text()) if cfg_path and Path(cfg_path).exists() else None
        )
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    caps = AllCapabilities(
        mcp_servers=(McpBinding(name="builtin", command=(), tools=("alpha",)),),
    )
    OpenClawAgent(
        AgentConfig(
            target=str(tmp_path / "oc"),
            model="gemini-3.1-pro-preview",
            capabilities=caps,
        )
    ).run("p")
    assert captured["cfg_path"]
    assert captured["config"] == {
        "tools": {"codeMode": False, "deny": ["sessions_spawn", "sessions_yield"]},
        "memory": {"search": {"enabled": False}},
    }


def test_native_openai_key_crosses_explicit_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert _build_env(AgentConfig(provider="openai"))["OPENAI_API_KEY"] == "test-key"


@pytest.mark.parametrize(
    "model,provider",
    [("gemini-3.7-flash", "google-vertex"), ("claude-sonnet-5", "anthropic-vertex")],
)
def test_fleet_models_registered(model: str, provider: str) -> None:
    assert (
        _build_model_override(AgentConfig(model=model, provider=provider))["models"]["providers"][
            provider
        ]["models"][0]["id"]
        == model
    )


def test_vertex_auth_profile_seeded_for_headless_run() -> None:
    command = oc_mod._build_local_command(
        AgentConfig(provider="anthropic-vertex"), "hi", "operator", "oc"
    )
    assert "models auth paste-api-key --provider anthropic-vertex --agent operator" in command
    assert oc_mod._VERTEX_CREDENTIALS_MARKER in command


def test_vertex_auth_profile_seeded_for_keyless_google_vertex() -> None:
    # oc's per-agent auth store gates every provider, not just the plugin one:
    # a keyless google-vertex run aborts with the same ProviderAuthError until
    # the marker profile exists.
    command = oc_mod._build_local_command(
        AgentConfig(provider="google-vertex"), "hi", "operator", "oc"
    )
    assert "models auth paste-api-key --provider google-vertex --agent operator" in command


def test_vertex_auth_profile_skipped_for_a_keyed_run() -> None:
    # A real key is already in the config oc reads; seeding the marker on top
    # would shadow it.
    command = oc_mod._build_local_command(
        AgentConfig(provider="google-vertex", api_key="k"), "hi", "operator", "oc"
    )
    assert "paste-api-key" not in command


def test_vertex_auth_profile_skipped_for_a_non_vertex_provider() -> None:
    command = oc_mod._build_local_command(AgentConfig(provider="google"), "hi", "operator", "oc")
    assert "paste-api-key" not in command


def test_sandbox_vertex_overlay_uses_metadata_without_host_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/private/host.json")
    monkeypatch.setattr(oc_mod, "sandbox_credential_env", lambda spec, *, project=None: {})
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="anthropic-vertex"), tmp_path)
    assert overlay["GOOGLE_CLOUD_PROJECT"] == "test-project"
    assert overlay["ANTHROPIC_VERTEX_USE_GCP_METADATA"] == "1"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in overlay
    assert (tmp_path / "node-fetch-shim" / "register.mjs").is_file()


def test_sandbox_provider_env_vertex_injects_the_metadata_emulator_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Inside the sandbox there is no ADC and the real metadata endpoint is
    # blocked, so the backend's mint-and-inject recipe supplies the credential.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-a")
    seen: dict = {}

    def fake_recipe(spec, *, project=None):
        seen["backend"] = spec.backend
        seen["project"] = project
        return {
            "GCE_METADATA_HOST": "host.docker.internal:41235",
            "GCE_METADATA_IP": "host.docker.internal:41235",
            "METADATA_SERVER_DETECTION": "assume-present",
        }

    monkeypatch.setattr(oc_mod, "sandbox_credential_env", fake_recipe)
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="google-vertex"), tmp_path)

    assert seen == {"backend": "vertex", "project": "proj-a"}
    assert overlay["GCE_METADATA_HOST"] == "host.docker.internal:41235"
    assert overlay["GCE_METADATA_IP"] == "host.docker.internal:41235"
    assert overlay["METADATA_SERVER_DETECTION"] == "assume-present"
    # The recipe rides alongside the routing vars, it does not replace them.
    assert overlay["GOOGLE_CLOUD_PROJECT"] == "proj-a"


def test_sandbox_provider_env_anthropic_vertex_gets_the_recipe_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # anthropic-vertex shares the vertex backend, and the project falls back to
    # the GCP_PROJECT_ID spelling the forwarding block reads.
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GCP_PROJECT_ID", "proj-b")
    seen: dict = {}

    def fake_recipe(spec, *, project=None):
        seen["backend"] = spec.backend
        seen["project"] = project
        return {"GCE_METADATA_HOST": "host.docker.internal:41235"}

    monkeypatch.setattr(oc_mod, "sandbox_credential_env", fake_recipe)
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="anthropic-vertex"), tmp_path)

    assert seen == {"backend": "vertex", "project": "proj-b"}
    assert overlay["GCE_METADATA_HOST"] == "host.docker.internal:41235"
    assert overlay["ANTHROPIC_VERTEX_USE_GCP_METADATA"] == "1"


def test_sandbox_provider_env_non_vertex_asks_for_no_model_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A key-based provider carries its own credential across the boundary.
    monkeypatch.setattr(
        oc_mod,
        "sandbox_credential_env",
        lambda *a, **k: pytest.fail("sandbox_credential_env called for a key-based provider"),
    )
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="google"), tmp_path)
    assert not {"GCE_METADATA_HOST", "GCE_METADATA_IP", "METADATA_SERVER_DETECTION"} & set(overlay)


def test_sandbox_provider_env_keyed_vertex_skips_the_recipe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A sandboxed google-vertex run with an explicit key was a working
    # configuration before the recipe existed (the key rides
    # GOOGLE_CLOUD_API_KEY via _build_env); it must not start requiring
    # BENCH_VERTEX_SANDBOX_SA.
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-a")
    monkeypatch.setattr(
        oc_mod,
        "sandbox_credential_env",
        lambda *a, **k: pytest.fail("sandbox_credential_env called for a keyed run"),
    )
    overlay = oc_mod._sandbox_provider_env(
        AgentConfig(provider="google-vertex", api_key="k"), tmp_path
    )
    assert not {"GCE_METADATA_HOST", "GCE_METADATA_IP", "METADATA_SERVER_DETECTION"} & set(overlay)


def test_sandbox_provider_env_vertex_defaults_the_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # oc aborts with "Vertex AI requires a location" when neither spelling is
    # set on the host, which forwarding alone cannot fix.
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCP_VERTEX_LOCATION", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-a")
    monkeypatch.setattr(oc_mod, "sandbox_credential_env", lambda spec, *, project=None: {})
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="google-vertex"), tmp_path)
    assert overlay["GOOGLE_CLOUD_LOCATION"] == "global"


def test_sandbox_provider_env_vertex_keeps_an_explicit_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Assigning unconditionally is not a clobber: vertex_location() reads
    # GOOGLE_CLOUD_LOCATION first, so a forwarded value resolves to itself.
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-east5")
    monkeypatch.setattr(oc_mod, "sandbox_credential_env", lambda spec, *, project=None: {})
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="google-vertex"), tmp_path)
    assert overlay["GOOGLE_CLOUD_LOCATION"] == "us-east5"


def test_sandbox_provider_env_vertex_reads_the_alternate_location_spelling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.setenv("GCP_VERTEX_LOCATION", "europe-west4")
    monkeypatch.setattr(oc_mod, "sandbox_credential_env", lambda spec, *, project=None: {})
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="google-vertex"), tmp_path)
    assert overlay["GOOGLE_CLOUD_LOCATION"] == "europe-west4"


def test_sandbox_provider_env_keyed_vertex_still_gets_a_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The location is a routing fact, not a credential: a keyed run needs it
    # just as much, and skips only the emulator recipe.
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCP_VERTEX_LOCATION", raising=False)
    overlay = oc_mod._sandbox_provider_env(
        AgentConfig(provider="google-vertex", api_key="k"), tmp_path
    )
    assert overlay["GOOGLE_CLOUD_LOCATION"] == "global"


def test_sandbox_provider_env_non_vertex_invents_no_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A google-genai run resolves its endpoint from the API key, not a region.
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("GCP_VERTEX_LOCATION", raising=False)
    overlay = oc_mod._sandbox_provider_env(AgentConfig(provider="google"), tmp_path)
    assert "GOOGLE_CLOUD_LOCATION" not in overlay


def test_execute_unsandboxed_vertex_asks_for_no_model_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Flag off keeps its own credentials: the host process has ADC of its own,
    # so the recipe is not consulted and no emulator is started. (The auth
    # profile *is* seeded on this path -- oc's store gates it there too -- see
    # test_vertex_auth_profile_seeded_for_keyless_google_vertex.)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-a")
    monkeypatch.setattr(
        oc_mod,
        "sandbox_credential_env",
        lambda *a, **k: pytest.fail("sandbox_credential_env called on an unsandboxed run"),
    )
    captured: dict = {}

    def fake_bash(cmd, **kwargs):
        captured["env"] = kwargs.get("extra_env") or {}
        return _make_subprocess_result(stdout="ok", returncode=0)

    _install_oc_run(monkeypatch, fake_bash, _empty_sessions_run)
    cfg = AgentConfig(target=str(tmp_path / "oc"), provider="google-vertex")
    OpenClawAgent(cfg).run("p")
    assert not {"GCE_METADATA_HOST", "GCE_METADATA_IP", "METADATA_SERVER_DETECTION"} & set(
        captured["env"]
    )


@pytest.mark.parametrize(
    "model,provider,transport",
    [
        ("gemini-3.8-flash", "google", "google-generative-ai"),
        ("gemini-3.8-flash", "google-vertex", "google-vertex"),
        ("claude-fable-5-1", "anthropic-vertex", "anthropic-messages"),
    ],
)
def test_latest_models_have_per_run_catalog_and_transport(
    model: str, provider: str, transport: str
) -> None:
    override = _build_model_override(AgentConfig(model=model, provider=provider))
    entry = override["models"]["providers"][provider]
    assert entry["models"] == [{"id": model, "name": model}]
    assert entry["api"] == transport
    assert override["agents"]["defaults"]["models"] == {f"{provider}/{model}": {}}
