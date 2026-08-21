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

"""Unit tests for :mod:`devops_bench.agents.api.agent`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from devops_bench.agents import AGENTS, AgentConfig
from devops_bench.agents.api import agent as agent_mod
from devops_bench.agents.api.agent import (
    ApiAgent,
    extract_tokens,
    fold_trajectory,
    sum_tokens,
)
from devops_bench.agents.capabilities import (
    AgentRules,
    AllCapabilities,
    McpBinding,
    SkillBinding,
    SupportsMcp,
    SupportsRules,
    SupportsSkills,
)
from devops_bench.agents.result import TOKEN_BUCKETS
from devops_bench.models.base import LLMClient


def _mcp_caps(command: str = "server", *, tools: tuple[str, ...] = ()) -> AllCapabilities:
    """Helper: build capabilities that turn the API agent's MCP path on."""
    return AllCapabilities(
        mcp_servers=(McpBinding(name="test", command=tuple(command.split()), tools=tools),),
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _Turn:
    """One scripted response in :class:`_FakeLLMClient`.

    Attributes:
        text: ``get_text_content`` payload for this turn.
        calls: Function calls returned by ``extract_function_calls`` (each in
            the neutral ``{"name", "args", "id"}`` shape).
        usage: Optional duck-typed usage object surfaced on the raw response.
        usage_attr: Which attribute name to attach ``usage`` under. Defaults to
            ``"usage_metadata"`` (Google shape); set ``"usage"`` to exercise
            Anthropic / OpenAI / Ollama paths.
    """

    text: str
    calls: list[dict] = field(default_factory=list)
    usage: Any = None
    usage_attr: str = "usage_metadata"


class _FakeLLMClient(LLMClient):
    """Scripted :class:`LLMClient` that returns ``_Turn`` objects in order.

    Records every ``generate_content`` invocation and tracks whether
    ``format_tools`` was called so the agent's caller-formats-tools contract is
    asserted.
    """

    def __init__(self, turns: list[_Turn]) -> None:
        self._turns = list(turns)
        self.calls: list[dict] = []
        self.format_tools_calls: list[Any] = []

    async def generate_content(
        self,
        contents: list[dict[str, Any]],
        tools: Any,
        system_instruction: str | None,
    ) -> Any:
        if not self._turns:
            raise AssertionError("FakeLLMClient ran out of scripted turns")
        turn = self._turns.pop(0)
        self.calls.append(
            {
                "contents": [dict(msg) for msg in contents],
                "tools": tools,
                "system_instruction": system_instruction,
            }
        )
        response = SimpleNamespace(text=turn.text, calls=turn.calls)
        if turn.usage is not None:
            setattr(response, turn.usage_attr, turn.usage)
        return response

    def format_tools(self, mcp_tools: Any) -> Any:
        # Snapshot so tests can assert the agent does pre-format tools before
        # calling the loop.
        self.format_tools_calls.append(list(mcp_tools))
        return ("formatted", tuple(getattr(t, "name", "") for t in mcp_tools))

    def extract_function_calls(self, response: Any) -> list[dict]:
        return list(response.calls)

    def get_text_content(self, response: Any) -> str:
        return response.text


class _FakeMCPClient:
    """Stand-in for :class:`MCPClient` exposing only what the agent uses.

    Records every call so tests assert dispatch hit MCP (or skipped it).
    """

    def __init__(self, tools: list[Any] | None = None) -> None:
        self._tools = list(tools or [])
        self.calls: list[tuple[str, dict]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> _FakeMCPClient:
        self.entered = True
        return self

    async def __aexit__(self, *_a: Any) -> None:
        self.exited = True

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=self._tools)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        block = SimpleNamespace(text=f"mcp-result-of-{name}")
        return SimpleNamespace(content=[block])


# ---------------------------------------------------------------------------
# Registration & registry wiring
# ---------------------------------------------------------------------------


def test_api_agent_registered_under_canonical_key() -> None:
    assert AGENTS.get("api") is ApiAgent


def test_package_import_alone_registers_api_agent() -> None:
    """``import devops_bench.agents.api`` (no ``.agent`` submodule import) must
    register the harness — the documented consumer convention resolves via
    ``AGENTS.get`` after a package import. Runs in a fresh interpreter because
    this test module itself imports ``.agent``, which would mask a regression
    (review finding: the package ``__init__`` didn't import the module, so the
    registration decorator never ran)."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import devops_bench.agents.api
        from devops_bench.agents.base import AGENTS
        assert AGENTS.get("api").__name__ == "ApiAgent"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# fold_trajectory
# ---------------------------------------------------------------------------


def test_fold_trajectory_pairs_assistant_calls_with_tool_results() -> None:
    contents = [
        {"role": "user", "content": "g"},
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {"name": "alpha", "args": {"a": 1}, "id": "1"},
                {"name": "beta", "args": {"b": 2}, "id": "2"},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "name": "alpha", "content": "A-result"},
        {"role": "tool", "tool_call_id": "2", "name": "beta", "content": "B-result"},
        {"role": "assistant", "content": "all done"},
    ]
    assert fold_trajectory(contents) == [
        {"name": "alpha", "args": {"a": 1}, "result": "A-result", "status": "completed"},
        {"name": "beta", "args": {"b": 2}, "result": "B-result", "status": "completed"},
    ]


def test_fold_trajectory_marks_dispatcher_errors_as_error_status() -> None:
    # The dispatcher returns ``"Error: ..."`` for a failed tool call (matching
    # the agent's _build_dispatch contract) — folding must surface that as the
    # ``"error"`` status so metrics see the failure mode, not a clean call.
    contents = [
        {"role": "user", "content": "g"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "boom", "args": {}, "id": "x"}],
        },
        {"role": "tool", "tool_call_id": "x", "name": "boom", "content": "Error: kaboom"},
        {"role": "assistant", "content": "abort"},
    ]
    folded = fold_trajectory(contents)
    assert folded == [
        {"name": "boom", "args": {}, "result": "Error: kaboom", "status": "error"},
    ]


def test_fold_trajectory_leaves_unmatched_call_as_called_status() -> None:
    # If a result never lands (e.g. dispatch raised before appending), the call
    # entry survives with ``status="called"`` and ``result=None`` rather than
    # being silently dropped.
    contents = [
        {"role": "user", "content": "g"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "lost", "args": {}, "id": "9"}],
        },
    ]
    assert fold_trajectory(contents) == [
        {"name": "lost", "args": {}, "result": None, "status": "called"},
    ]


def test_fold_trajectory_skips_text_only_turns() -> None:
    contents = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "bye"},
    ]
    assert fold_trajectory(contents) == []


def test_fold_trajectory_handles_none_args_and_none_call_id() -> None:
    # Defensive: missing ``args`` becomes ``{}``; an entry with no ``id`` is
    # emitted as ``status="called"`` (no result to pair with).
    contents = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "t", "args": None, "id": None}],
        },
    ]
    assert fold_trajectory(contents) == [
        {"name": "t", "args": {}, "result": None, "status": "called"},
    ]


def test_fold_trajectory_pairs_unkeyed_gemini_style_calls_fifo() -> None:
    """Gemini emits every function call with ``id=None`` and ``run_tool_loop``
    appends one result per call in order — the fold must pair them FIFO
    instead of dropping every result as an orphan (review finding: the default
    provider produced all-'called' trajectories and errored every clean run)."""
    contents = [
        {"role": "user", "content": "g"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "alpha", "args": {"a": 1}, "id": None},
                {"name": "beta", "args": {"b": 2}, "id": None},
            ],
        },
        {"role": "tool", "tool_call_id": None, "name": "alpha", "content": "A-result"},
        {"role": "tool", "tool_call_id": None, "name": "beta", "content": "Error: b"},
        {"role": "assistant", "content": "done"},
    ]
    from devops_bench.agents.api.agent import _fold_with_extraction_errors

    folded, orphans = _fold_with_extraction_errors(contents)
    assert orphans == []
    assert folded == [
        {"name": "alpha", "args": {"a": 1}, "result": "A-result", "status": "completed"},
        {"name": "beta", "args": {"b": 2}, "result": "Error: b", "status": "error"},
    ]


def test_fold_trajectory_extra_unkeyed_result_is_still_an_orphan() -> None:
    """Id-less results beyond the number of id-less calls stay orphans —
    FIFO pairing must not absorb genuinely unmatched results."""
    contents = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "t", "args": {}, "id": None}],
        },
        {"role": "tool", "tool_call_id": None, "name": "t", "content": "paired"},
        {"role": "tool", "tool_call_id": None, "name": "ghost", "content": "stray"},
    ]
    from devops_bench.agents.api.agent import _fold_with_extraction_errors

    folded, orphans = _fold_with_extraction_errors(contents)
    assert folded == [{"name": "t", "args": {}, "result": "paired", "status": "completed"}]
    assert len(orphans) == 1
    assert "stray" in orphans[0]


def test_fold_trajectory_drops_unpaired_tool_results_silently_from_trajectory() -> None:
    """An orphan ``role: tool`` (no matching assistant call_id) is dropped from
    the canonical trajectory — synthesizing a free-floating entry would break
    the "every trajectory item is a real ToolCall the model issued" invariant
    metrics depend on. The diagnostic flows out via ``_fold_with_extraction_errors``
    instead (asserted in ``test_execute_orphan_tool_result_lands_in_errors``).
    """
    contents = [
        {"role": "user", "content": "g"},
        # No assistant turn → no matching call_id for the ghost result.
        {"role": "tool", "tool_call_id": "ghost", "name": "x", "content": "?"},
    ]
    assert fold_trajectory(contents) == []


def test_fold_with_extraction_errors_surfaces_orphan_results() -> None:
    from devops_bench.agents.api.agent import _fold_with_extraction_errors

    contents = [
        {"role": "user", "content": "g"},
        {"role": "tool", "tool_call_id": "ghost", "name": "x", "content": "stray"},
    ]
    folded, orphans = _fold_with_extraction_errors(contents)
    assert folded == []
    assert len(orphans) == 1
    assert "no matching call" in orphans[0]
    assert "ghost" in orphans[0]


# ---------------------------------------------------------------------------
# extract_tokens
# ---------------------------------------------------------------------------


def test_extract_tokens_emits_the_canonical_buckets() -> None:
    """The API harness reports the same six buckets as gemini_cli / antigravity.

    It previously emitted only ``prompt_tokens`` / ``candidates_tokens`` /
    ``total_tokens``, so every API row had ``cached``, ``cacheWrite`` and
    ``reasoning`` as null and no cache-discounted cost could be computed.
    """
    usage = SimpleNamespace(prompt_token_count=3, candidates_token_count=5, total_token_count=8)
    response = SimpleNamespace(usage_metadata=usage)
    assert set(extract_tokens(response)) == set(TOKEN_BUCKETS)


def test_extract_tokens_falls_back_to_usage_attribute() -> None:
    usage = SimpleNamespace(prompt_token_count=1, candidates_token_count=2, total_token_count=3)
    # No ``usage_metadata``; the function should still find ``usage``.
    response = SimpleNamespace(usage=usage)
    assert extract_tokens(response)["total"] == 3


def test_extract_tokens_returns_empty_dict_when_no_usage() -> None:
    assert extract_tokens(SimpleNamespace()) == {}
    assert extract_tokens(None) == {}


def test_extract_tokens_returns_empty_dict_for_an_unknown_usage_shape() -> None:
    """A usage object in a shape we cannot map reads as unavailable, not free.

    Zeros would claim the turn cost nothing, which is indistinguishable from a
    genuinely empty turn and silently understates a new provider's cost.
    """
    usage = SimpleNamespace(some_future_field=12)
    assert extract_tokens(SimpleNamespace(usage=usage)) == {}


def test_extract_tokens_unreported_buckets_are_none_not_zero() -> None:
    """A provider that sends no cache telemetry yields ``None``, not ``0``.

    ``0`` would read downstream as "nothing was cached" when the truth is
    "this provider never says".
    """
    usage = SimpleNamespace(prompt_token_count=4, candidates_token_count=1)
    tokens = extract_tokens(SimpleNamespace(usage_metadata=usage))
    assert tokens["cached"] is None
    assert tokens["cache_write"] is None
    assert tokens["reasoning"] is None
    assert tokens["input"] == 4
    assert tokens["output"] == 1


def test_extract_tokens_google_subtracts_cached_from_the_prompt() -> None:
    """Google's ``prompt_token_count`` includes cached tokens.

    Canonical ``input`` is the non-cached prompt, so counting the prompt
    verbatim beside ``cached`` would bill the cached tokens twice — once at the
    full input rate and once at the discounted one.
    """
    usage = SimpleNamespace(
        prompt_token_count=1000,
        cached_content_token_count=800,
        candidates_token_count=50,
        thoughts_token_count=30,
        total_token_count=1080,
    )
    assert extract_tokens(SimpleNamespace(usage_metadata=usage)) == {
        "input": 200,
        "cached": 800,
        "cache_write": None,
        "reasoning": 30,
        "output": 50,
        "total": 1080,
    }


def test_extract_tokens_anthropic_keeps_input_and_adds_cache_buckets() -> None:
    """Anthropic's ``input_tokens`` already excludes cache reads and writes.

    It is the one shape here that must *not* be adjusted; subtracting would
    undercount the billed prompt.
    """
    usage = SimpleNamespace(
        input_tokens=42,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=100,
        output_tokens=17,
    )
    assert extract_tokens(SimpleNamespace(usage=usage)) == {
        "input": 42,
        "cached": 900,
        "cache_write": 100,
        # Thinking is billed inside output_tokens and never reported apart.
        "reasoning": None,
        "output": 17,
        "total": 1059,
    }


def test_extract_tokens_anthropic_total_no_longer_drops_cache_tokens() -> None:
    """Regression: the old ``input + output`` fallback lost every cache token.

    Anthropic reports no aggregated total. On a cached agentic run the cache
    buckets are most of the spend, so the old total under-reported this turn as
    59 tokens against a real 1059.
    """
    usage = SimpleNamespace(
        input_tokens=42,
        cache_read_input_tokens=900,
        cache_creation_input_tokens=100,
        output_tokens=17,
    )
    assert extract_tokens(SimpleNamespace(usage=usage))["total"] == 1059


def test_extract_tokens_openai_reads_nested_detail_objects() -> None:
    """OpenAI nests cached and reasoning counts inside their parent count.

    Both are *included* in the parent, so each is subtracted back out to keep
    ``input`` non-cached and ``output`` reasoning-free.
    """
    usage = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_tokens_details=SimpleNamespace(cached_tokens=600),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=150),
    )
    assert extract_tokens(SimpleNamespace(usage=usage)) == {
        "input": 400,
        "cached": 600,
        "cache_write": None,
        "reasoning": 150,
        "output": 50,
        "total": 1200,
    }


def test_extract_tokens_ollama_shape_without_detail_objects() -> None:
    """Ollama omits both detail objects; the buckets they feed stay ``None``."""
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert extract_tokens(SimpleNamespace(usage=usage)) == {
        "input": 10,
        "cached": None,
        "cache_write": None,
        "reasoning": None,
        "output": 20,
        "total": 30,
    }


def test_extract_tokens_openai_shim_attributes_the_total_shortfall_to_reasoning() -> None:
    """A shim can bill thinking into the total but omit ``completion_tokens_details``.

    Observed live against Gemini's OpenAI-compatible endpoint: ``prompt=6,
    completion=1, total=61``. Without this the 54 thinking tokens land in no
    bucket, so the row's buckets under-sum its own total by 88%.
    """
    usage = SimpleNamespace(prompt_tokens=6, completion_tokens=1, total_tokens=61)
    buckets = extract_tokens(SimpleNamespace(usage=usage))
    assert buckets == {
        "input": 6,
        "cached": None,
        "cache_write": None,
        "reasoning": 54,
        "output": 1,
        "total": 61,
    }
    assert sum(v for k, v in buckets.items() if k != "total" and v is not None) == 61


def test_extract_tokens_openai_reported_reasoning_is_not_re_derived() -> None:
    """A reported ``reasoning_tokens`` wins; the shortfall rule must not fire.

    Real OpenAI bills reasoning *inside* ``completion_tokens``, so
    ``total == prompt + completion`` and deriving a second time would both
    double-count and wrongly shrink ``output``.
    """
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=80,
        total_tokens=180,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=60),
    )
    assert extract_tokens(SimpleNamespace(usage=usage)) == {
        "input": 100,
        "cached": None,
        "cache_write": None,
        "reasoning": 60,
        "output": 20,
        "total": 180,
    }


def test_extract_tokens_openai_no_shortfall_leaves_reasoning_unreported() -> None:
    """A non-reasoning model has no gap, so ``reasoning`` stays ``None``.

    Verified live: ``gemini-3.1-flash-lite`` through the same shim reports a
    gap of 0 where the reasoning model reports 54.
    """
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert extract_tokens(SimpleNamespace(usage=usage))["reasoning"] is None


def test_extract_tokens_openai_negative_shortfall_is_ignored() -> None:
    """A total below the reported counts is a provider bug, not negative thinking."""
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=120)
    buckets = extract_tokens(SimpleNamespace(usage=usage))
    assert buckets["reasoning"] is None
    assert buckets["output"] == 50


def test_extract_tokens_provider_total_wins_over_the_bucket_sum() -> None:
    """A provider-supplied total is passed through, never second-guessed."""
    usage = SimpleNamespace(prompt_token_count=5, candidates_token_count=7, total_token_count=99)
    assert extract_tokens(SimpleNamespace(usage_metadata=usage))["total"] == 99


def test_extract_tokens_a_reported_zero_total_falls_back_to_the_bucket_sum() -> None:
    """An unset protobuf int reads as 0; billed buckets must not sum to it."""
    usage = SimpleNamespace(
        prompt_token_count=4000, candidates_token_count=120, total_token_count=0
    )
    assert extract_tokens(SimpleNamespace(usage_metadata=usage))["total"] == 4120


def test_extract_tokens_google_over_reported_cached_cannot_go_negative() -> None:
    """``cached`` above the prompt count clamps ``input`` at 0, never below."""
    usage = SimpleNamespace(prompt_token_count=100, cached_content_token_count=150)
    assert extract_tokens(SimpleNamespace(usage_metadata=usage))["input"] == 0


def test_extract_tokens_openai_reasoning_above_completion_cannot_go_negative() -> None:
    """A shim billing reasoning *outside* ``completion_tokens`` must not yield
    a negative ``output`` that then subtracts from the run's total."""
    usage = SimpleNamespace(
        prompt_tokens=6,
        completion_tokens=1,
        total_tokens=61,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=54),
    )
    assert extract_tokens(SimpleNamespace(usage=usage))["output"] == 0


def test_extract_tokens_computes_the_total_from_every_bucket() -> None:
    """With no provider total, the sum spans all five buckets, not just two."""
    usage = SimpleNamespace(
        input_tokens=1,
        cache_read_input_tokens=2,
        cache_creation_input_tokens=4,
        output_tokens=8,
    )
    assert extract_tokens(SimpleNamespace(usage=usage))["total"] == 15


# ---------------------------------------------------------------------------
# sum_tokens
# ---------------------------------------------------------------------------


def _usage_response(prompt: int, candidates: int, total: int) -> SimpleNamespace:
    """Build a Google-shaped response carrying the given counts."""
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            total_token_count=total,
        )
    )


def test_sum_tokens_adds_every_turn() -> None:
    """A run's usage is the sum over turns, not the last turn's counts.

    Regression test: the agent read ``LoopResult.response`` — the final turn —
    so a multi-turn run reported a fraction of what it was billed, and the
    shortfall grew with turn count.
    """
    responses = [
        _usage_response(100, 10, 110),
        _usage_response(300, 20, 320),
        _usage_response(700, 5, 705),
    ]
    assert sum_tokens(responses)["input"] == 1100
    assert sum_tokens(responses)["output"] == 35
    assert sum_tokens(responses)["total"] == 1135
    # The old behaviour, for contrast: the last turn alone.
    assert extract_tokens(responses[-1])["total"] == 705


def test_sum_tokens_skips_turns_that_report_no_usage() -> None:
    """A turn with no usage block contributes nothing rather than zeroing the sum."""
    responses = [_usage_response(5, 1, 6), SimpleNamespace(), None, _usage_response(7, 2, 9)]
    assert sum_tokens(responses)["input"] == 12
    assert sum_tokens(responses)["output"] == 3
    assert sum_tokens(responses)["total"] == 15


def test_sum_tokens_keeps_a_bucket_none_when_no_turn_reported_it() -> None:
    """``None`` must not decay to ``0`` just because turns were summed.

    Google sends no ``cache_write`` count, so a summed run that shows ``0``
    there would claim the run wrote nothing to cache rather than admitting the
    provider never said.
    """
    assert sum_tokens([_usage_response(5, 1, 6), _usage_response(7, 2, 9)])["cache_write"] is None


def test_sum_tokens_treats_a_silent_turn_as_zero_once_any_turn_reports() -> None:
    """One turn reporting a bucket is enough to make the run's figure a number.

    A cache hit on turn two is real spend even though turn one predates the
    cache, so the sum is that turn's count, not ``None``.
    """
    cached_turn = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=900,
            cached_content_token_count=800,
            candidates_token_count=5,
        )
    )
    assert sum_tokens([_usage_response(100, 10, 110), cached_turn])["cached"] == 800


def test_sum_tokens_returns_empty_dict_when_nothing_reported() -> None:
    """No turns, or no turn with usage, yields ``{}`` — the same signal
    :func:`extract_tokens` gives, so ``results.json`` omits the block rather
    than recording a run that used zero tokens."""
    assert sum_tokens([]) == {}
    assert sum_tokens([SimpleNamespace(), None]) == {}


def test_sum_tokens_single_turn_matches_extract_tokens() -> None:
    """A one-turn run must report exactly what the single turn reported."""
    response = _usage_response(11, 22, 33)
    assert sum_tokens([response]) == extract_tokens(response)


# ---------------------------------------------------------------------------
# ApiAgent._execute — MCP-off path
# ---------------------------------------------------------------------------


def test_execute_runs_with_no_tools_when_capabilities_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default capabilities (no MCP binding, no skills) → loop runs tool-less.

    Renamed from ``..._when_target_unset`` because the MCP gate is no longer
    ``config.target`` — it's ``config.capabilities.mcp``. The old name
    survives in git history but the new one matches the post-PR3 reality.
    """
    fake = _FakeLLMClient(
        [
            _Turn(
                text="done",
                usage=SimpleNamespace(
                    prompt_token_count=3,
                    candidates_token_count=5,
                    total_token_count=8,
                ),
            )
        ]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)

    agent = ApiAgent(AgentConfig())
    result = agent.run("hello")

    assert result.output == "done"
    assert result.trajectory == []
    assert result.tokens == {
        "input": 3,
        "cached": None,
        "cache_write": None,
        "reasoning": None,
        "output": 5,
        "total": 8,
    }
    assert result.errors == []
    # Caller-formats-tools: the agent must call format_tools on the (empty)
    # skill list before invoking the loop.
    assert fake.format_tools_calls == [[]]
    # The loop saw the pre-formatted tools sentinel, not the raw list.
    assert fake.calls[0]["tools"] == ("formatted", ())


def test_execute_passes_explicit_provider_and_model_to_get_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_get_model(provider, model):
        captured["provider"] = provider
        captured["model"] = model
        return _FakeLLMClient([_Turn(text="ok")])

    monkeypatch.setattr(agent_mod, "get_model", fake_get_model)
    cfg = AgentConfig(provider="anthropic", model="claude-3-5")
    ApiAgent(cfg).run("p")
    assert captured == {"provider": "anthropic", "model": "claude-3-5"}


def test_execute_records_anthropic_tokens_through_to_agentresult(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an Anthropic-shaped usage object surfaces on
    ``AgentResult.tokens`` in the canonical buckets — regression for the
    blocking bug where non-Google providers logged zero tokens with no error."""
    fake = _FakeLLMClient(
        [
            _Turn(
                text="done",
                usage=SimpleNamespace(
                    input_tokens=42,
                    cache_read_input_tokens=900,
                    output_tokens=17,
                ),
                usage_attr="usage",
            )
        ]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    result = ApiAgent(AgentConfig()).run("p")
    assert result.tokens == {
        "input": 42,
        "cached": 900,
        "cache_write": None,
        "reasoning": None,
        "output": 17,
        "total": 959,
    }


def test_execute_records_one_turn_entry_per_provider_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run's totals hide their own shape.

    40k tokens over 3 turns and 40k over 30 produce the same ``tokens`` dict,
    and only the second is a loop failing to converge, so the per-turn split
    has to reach ``AgentResult``.
    """
    fake = _FakeLLMClient(
        [
            _Turn(
                text="working",
                calls=[{"name": "t", "args": {}, "id": "a"}],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                usage_attr="usage",
            ),
            _Turn(
                text="done",
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=7, total_tokens=37),
                usage_attr="usage",
            ),
        ]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    result = ApiAgent(AgentConfig()).run("p")

    assert [turn["tool_calls"] for turn in result.turns] == [1, 0]
    assert [turn["tokens"]["total"] for turn in result.turns] == [15, 37]
    assert result.tokens["total"] == 52
    assert sum(turn["latency_sec"] for turn in result.turns) == pytest.approx(result.latency)


def test_execute_leaves_turns_empty_when_the_loop_never_ran(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No turn taken means no turn measured — not a turn of zeros."""
    fake = _FakeLLMClient([])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    assert ApiAgent(AgentConfig(max_turns=0)).run("p").turns == []


def test_execute_records_openai_tokens_through_to_agentresult(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an OpenAI/Ollama-shaped usage object surfaces in the
    canonical buckets."""
    fake = _FakeLLMClient(
        [
            _Turn(
                text="done",
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                usage_attr="usage",
            )
        ]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    result = ApiAgent(AgentConfig()).run("p")
    assert result.tokens == {
        "input": 10,
        "cached": None,
        "cache_write": None,
        "reasoning": None,
        "output": 20,
        "total": 30,
    }


# ---------------------------------------------------------------------------
# ApiAgent._execute — MCP-on, trajectory folding & tools_used metadata
# ---------------------------------------------------------------------------


def test_execute_folds_assistant_tool_pairs_into_canonical_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an MCP tool turn followed by a finishing turn → one ToolCall."""
    fc = [{"name": "do_thing", "args": {"a": 1}, "id": "call-1"}]
    fake = _FakeLLMClient(
        [
            _Turn(text="working", calls=fc),
            _Turn(
                text="done",
                usage=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=20,
                    total_token_count=30,
                ),
            ),
        ]
    )
    mcp_advertised = [SimpleNamespace(name="do_thing", description="d", inputSchema=None)]
    mcp = _FakeMCPClient(tools=mcp_advertised)
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", lambda _path: mcp)

    result = ApiAgent(AgentConfig(capabilities=_mcp_caps("server"))).run("ping")

    assert result.output == "done"
    assert result.trajectory == [
        {
            "name": "do_thing",
            "args": {"a": 1},
            "result": "mcp-result-of-do_thing",
            "status": "completed",
        },
    ]
    assert result.tokens == {
        "input": 10,
        "cached": None,
        "cache_write": None,
        "reasoning": None,
        "output": 20,
        "total": 30,
    }
    assert result.errors == []
    assert result.metadata["tools_used"] == ["do_thing"]
    # MCP session was entered & exited via the async-context-manager protocol.
    assert mcp.entered and mcp.exited
    # The agent passed the MCP-advertised tool through format_tools before
    # invoking the loop (caller-formats-tools contract).
    assert fake.format_tools_calls == [mcp_advertised]


# ---------------------------------------------------------------------------
# Dispatcher error handling
# ---------------------------------------------------------------------------


def test_execute_dispatch_error_lands_in_errors_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool call that raises must surface on errors, not crash the agent."""

    class _ExplodingMCP(_FakeMCPClient):
        async def call_tool(self, name: str, arguments: dict) -> Any:
            raise RuntimeError("kaboom")

    fc = [{"name": "boom", "args": {}, "id": "c1"}]
    fake = _FakeLLMClient(
        [
            _Turn(text="trying", calls=fc),
            _Turn(text="giving up"),
        ]
    )
    mcp = _ExplodingMCP(tools=[SimpleNamespace(name="boom", description="d", inputSchema=None)])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", lambda _path: mcp)

    result = ApiAgent(AgentConfig(capabilities=_mcp_caps("server"))).run("ping")

    assert result.output == "giving up"
    assert any("Error calling tool boom" in e for e in result.errors)
    # The trajectory still records the failed call with ``status="error"`` so
    # the metrics layer can see the failure mode.
    assert result.trajectory == [
        {"name": "boom", "args": {}, "result": "Error: kaboom", "status": "error"},
    ]


def test_execute_marks_mcp_iserror_result_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MCP server reporting failure via ``CallToolResult.isError`` (rather
    than raising) must fold as ``status="error"`` and land on ``errors`` —
    previously the flag was ignored and failures scored as completed (review
    finding)."""

    class _IsErrorMCP(_FakeMCPClient):
        async def call_tool(self, name: str, arguments: dict) -> Any:
            block = SimpleNamespace(text="unknown pod foo")
            return SimpleNamespace(content=[block], isError=True)

    fc = [{"name": "get_pod", "args": {}, "id": "c1"}]
    fake = _FakeLLMClient([_Turn(text="trying", calls=fc), _Turn(text="giving up")])
    mcp = _IsErrorMCP(tools=[SimpleNamespace(name="get_pod", description="d", inputSchema=None)])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", lambda _path: mcp)

    result = ApiAgent(AgentConfig(capabilities=_mcp_caps("server"))).run("ping")

    assert any("Error calling tool get_pod" in e for e in result.errors)
    assert result.trajectory == [
        {
            "name": "get_pod",
            "args": {},
            "result": "Error: unknown pod foo",
            "status": "error",
        },
    ]


def test_execute_records_missing_mcp_when_tool_requested_without_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No MCP server, but the model still requests a tool → recorded, not crashed."""
    fc = [{"name": "ghost", "args": {}, "id": "c2"}]
    fake = _FakeLLMClient(
        [
            _Turn(text="requesting", calls=fc),
            _Turn(text="abandoned"),
        ]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    result = ApiAgent(AgentConfig()).run("p")  # no target -> no MCP

    assert any("no MCP server is configured" in e for e in result.errors)
    assert result.trajectory == [
        {
            "name": "ghost",
            "args": {},
            "result": (
                "Error: tool 'ghost' requested but no MCP server is configured for this agent."
            ),
            "status": "error",
        },
    ]


# ---------------------------------------------------------------------------
# Skills are independent of MCP
# ---------------------------------------------------------------------------


def test_execute_skills_discover_without_mcp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Skills must be loadable on the MCP-off path, not gated on MCP being on."""
    skill_dir = tmp_path / "skills"
    skill = skill_dir / "demo" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill_body = '---\nname: "demo-skill"\ndescription: does things\n---\nbody\n'
    skill.write_text(skill_body)

    fc = [{"name": "skill_demo_skill", "args": {}, "id": "s1"}]
    fake = _FakeLLMClient(
        [
            _Turn(text="using skill", calls=fc),
            _Turn(text="done"),
        ]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    cfg = AgentConfig(
        capabilities=AllCapabilities(skills=SkillBinding(paths=(str(skill_dir),))),
    )  # NB: no mcp binding → MCP path disabled
    result = ApiAgent(cfg).run("p")

    assert result.errors == []  # skill dispatch hit the file, not the MCP error
    # Dispatch serves the whole file (frontmatter + body) captured at
    # discovery so the
    # model can read either as it sees fit — matches the legacy semantic.
    assert result.trajectory == [
        {
            "name": "skill_demo_skill",
            "args": {},
            "result": skill_body,
            "status": "completed",
        },
    ]
    assert result.metadata["skills_loaded"] == ["demo-skill"]
    # format_tools received the synthetic skill descriptor — one entry.
    assert len(fake.format_tools_calls[0]) == 1
    assert fake.format_tools_calls[0][0].name == "skill_demo_skill"


def test_execute_no_skills_no_mcp_runs_tool_less(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty skills + no target → loop seen no tools, no metadata noise."""
    fake = _FakeLLMClient([_Turn(text="hi")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    result = ApiAgent(AgentConfig()).run("p")
    assert result.output == "hi"
    assert "skills_loaded" not in result.metadata
    assert result.metadata["tools_used"] == []


# ---------------------------------------------------------------------------
# Surfaced model-construction errors / empty MCP target
# ---------------------------------------------------------------------------


def test_execute_lets_value_error_reach_base_safety_net(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``ValueError`` raised inside the run (MCP setup, provider adapter,
    anywhere) must bubble to :meth:`AgentHarness.run`'s safety net, which logs
    the full traceback and converts to an errored result tagged with the
    exception type.

    The agent previously intercepted ``ValueError`` itself "for empty MCP
    commands" — a case its own pre-guard makes unreachable — thereby swallowing
    unrelated ValueErrors without a traceback (review finding); the catch is
    gone.
    """

    class _BoomMCP:
        def __init__(self, _path: str) -> None: ...

        async def __aenter__(self) -> _BoomMCP:
            raise ValueError("bad provider argument")

        async def __aexit__(self, *_a: Any) -> None: ...

    fake = _FakeLLMClient([_Turn(text="never reached")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", _BoomMCP)

    caps = AllCapabilities(
        mcp_servers=(McpBinding(name="bad", command=("/never-spawned",)),),
    )
    result = ApiAgent(AgentConfig(capabilities=caps)).run("p")
    assert result.has_errors()
    # The base safety net formats as "<ExcType>: <msg>" — proof it was used.
    assert result.errors[0] == "ValueError: bad provider argument"
    assert result.output.startswith("Error: ")


def test_execute_times_out_via_config_timeout_sec(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AgentConfig.timeout_sec`` must bound the whole run — a hanging MCP
    server or provider call converts to an errored result at the deadline
    instead of wedging the benchmark (review finding: the field was ignored)."""

    async def _hang(*_a: Any, **_kw: Any):
        await asyncio.sleep(30)

    monkeypatch.setattr(agent_mod, "_run_async", _hang)
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: _FakeLLMClient([]))

    result = ApiAgent(AgentConfig(timeout_sec=0.05)).run("p")
    assert result.has_errors()
    assert "timed out after 0.05s" in result.errors[0]


def test_execute_reraises_internal_timeout_before_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``TimeoutError`` from inside the run (e.g. a provider socket timeout —
    ``socket.timeout`` is ``TimeoutError`` since 3.10) raised well before the
    configured deadline must not be relabeled as the agent-level timeout: it
    re-raises to the base safety net, which tags the exception type and logs
    the traceback."""

    async def _boom(*_a: Any, **_kw: Any):
        raise TimeoutError("socket read timed out")

    monkeypatch.setattr(agent_mod, "_run_async", _boom)
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: _FakeLLMClient([]))

    result = ApiAgent(AgentConfig(timeout_sec=600.0)).run("p")
    assert result.has_errors()
    assert result.errors[0] == "TimeoutError: socket read timed out"


def test_execute_honors_max_turns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_turns=0`` means zero model turns — it must not be silently
    replaced by the 50-turn default via a falsy-``or`` (review finding)."""
    fake = _FakeLLMClient([_Turn(text="never called")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    result = ApiAgent(AgentConfig(max_turns=0)).run("p")
    # The loop never invoked the model.
    assert fake.calls == []
    assert result.output == ""


def test_execute_warns_when_extra_mcp_servers_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the first MCP binding is honored (documented aggregate behavior);
    dropping servers 2..N must at least be visible in the logs."""
    fake = _FakeLLMClient([_Turn(text="ok")])
    mcp = _FakeMCPClient(tools=[])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", lambda _path: mcp)
    caps = AllCapabilities(
        mcp_servers=(
            McpBinding(name="one", command=("server-a",)),
            McpBinding(name="two", command=("server-b",)),
        ),
    )
    with caplog.at_level("WARNING", logger="devops_bench.agents.api.agent"):
        ApiAgent(AgentConfig(capabilities=caps)).run("p")
    assert any("only the first MCP binding" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# No env-smuggling — neither BENCH_USE_MCP nor any direct env read
# ---------------------------------------------------------------------------


def test_execute_ignores_bench_use_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting BENCH_USE_MCP must not change the agent's behavior.

    The MCP on/off gate is ``config.capabilities.mcp`` (a binding with a
    non-empty ``command``), not env.
    """
    monkeypatch.setenv("BENCH_USE_MCP", "false")
    fake = _FakeLLMClient([_Turn(text="ok")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    # No MCP binding → loop runs without MCP regardless of the env flag.
    result = ApiAgent(AgentConfig()).run("p")
    assert result.output == "ok"


def test_agent_source_has_no_bench_use_mcp_or_environ_reads() -> None:
    """Statically verify the agents/api source carries no env-smuggling.

    The conventions doc forbids agents from reading ``BENCH_USE_MCP`` (the
    harness threads the boolean instead) or any ``os.environ`` lookups inside
    the agent — capability/MCP on-off comes from ``AgentConfig``. This test
    walks the agents/api source AST and asserts no code (i.e. anything that is
    not a docstring or comment) references those names.
    """
    import ast
    import pathlib

    api_root = pathlib.Path(agent_mod.__file__).parent
    forbidden_names = {"get_env", "get_bool", "first_env", "require_env"}
    forbidden_strings = {"BENCH_USE_MCP"}

    for src in api_root.glob("*.py"):
        tree = ast.parse(src.read_text(), filename=str(src))

        for node in ast.walk(tree):
            # Bare names: ``get_env(...)`` / ``os.environ``.
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                pytest.fail(
                    f"{src.name} references env-helper {node.id!r} at line "
                    f"{node.lineno}; config flows through AgentConfig"
                )
            # Attribute access: ``os.environ``, ``os.getenv``, and the
            # ``os.environ.get(...)`` call form (which presents as an outer
            # Attribute("get", Attribute("environ", Name("os")))).
            if isinstance(node, ast.Attribute):
                attr_chain = []
                cur: ast.AST | None = node
                while isinstance(cur, ast.Attribute):
                    attr_chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    attr_chain.append(cur.id)
                joined = ".".join(reversed(attr_chain))
                assert joined != "os.environ", (
                    f"{src.name} accesses os.environ at line {node.lineno}; "
                    "config flows through AgentConfig"
                )
                assert joined != "os.getenv", (
                    f"{src.name} calls os.getenv at line {node.lineno}; "
                    "config flows through AgentConfig"
                )
                assert joined != "os.environ.get", (
                    f"{src.name} calls os.environ.get at line {node.lineno}; "
                    "config flows through AgentConfig"
                )
            # Subscript access: ``os.environ["FOO"]`` / ``os.environ.get(...)``
            # patterns. The outer ``ast.Subscript`` wraps the ``os.environ``
            # attribute lookup; the inner Attribute is also caught above via
            # ast.walk, but checking the Subscript explicitly makes the intent
            # legible and bug-proofs the guard against future AST refactors.
            if isinstance(node, ast.Subscript):
                target = node.value
                if isinstance(target, ast.Attribute):
                    attr_chain = []
                    cur = target
                    while isinstance(cur, ast.Attribute):
                        attr_chain.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name):
                        attr_chain.append(cur.id)
                    joined = ".".join(reversed(attr_chain))
                    assert joined != "os.environ", (
                        f"{src.name} subscripts os.environ[...] at line "
                        f"{node.lineno}; config flows through AgentConfig"
                    )
            # String literals: catch ``getenv("BENCH_USE_MCP")`` /
            # ``os.environ["BENCH_USE_MCP"]`` patterns even when wrapped in a
            # helper we didn't enumerate. Docstrings are still ``Str``/
            # ``Constant`` nodes but the parent statement is an ``Expr`` at the
            # head of a module/function/class — skip those.
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in forbidden_strings
            ):
                # Allow doc references — only reject when the same line names an
                # env accessor (``environ`` / ``getenv`` / ``get_env``).
                line_text = src.read_text().splitlines()[node.lineno - 1]
                accessor_hit = any(a in line_text for a in ("environ", "getenv", "get_env"))
                assert not accessor_hit, (
                    f"{src.name} reads {node.value!r} via an env accessor at line {node.lineno}"
                )


def test_api_package_does_not_expose_legacy_context_or_system_instruction() -> None:
    """The PR2 contract drops the ``context``/``system_instruction`` grab-bag.

    No symbol or kwarg with that name should remain in the public surface.
    """
    import inspect

    sig = inspect.signature(ApiAgent.run)
    assert list(sig.parameters) == ["self", "prompt", "workspace_path"], (
        "ApiAgent.run must take only (self, prompt, workspace_path); the legacy "
        "context/system_instruction kwargs are gone."
    )
    assert "context" not in sig.parameters
    assert "system_instruction" not in sig.parameters
    init_sig = inspect.signature(ApiAgent.__init__)
    # Inherited from AgentHarness — accepts config only.
    assert list(init_sig.parameters) == ["self", "config"]


@pytest.mark.parametrize("method_name", ["_execute"])
def test_execute_is_synchronous_and_safe_for_harness_invocation(method_name: str) -> None:
    """The harness calls ``run(prompt)`` synchronously; ``_execute`` must be sync too."""
    import inspect

    method = getattr(ApiAgent, method_name)
    assert not inspect.iscoroutinefunction(method), (
        f"ApiAgent.{method_name} must be synchronous so the harness can call "
        "agent.run(prompt) without managing an event loop itself."
    )


# ---------------------------------------------------------------------------
# PR3 — capability negotiation: ApiAgent implements every Supports* Protocol
# ---------------------------------------------------------------------------


def test_api_agent_satisfies_all_three_capability_protocols() -> None:
    """``isinstance`` against each Protocol is how the harness negotiates
    capabilities before granting a binding (handoff §6). ApiAgent declares
    every capability it can drive — MCP, skills, rules — so an instance must
    pass each ``runtime_checkable`` check."""
    agent = ApiAgent(AgentConfig())
    assert isinstance(agent, SupportsMcp)
    assert isinstance(agent, SupportsSkills)
    assert isinstance(agent, SupportsRules)


def test_api_agent_mirrors_capability_bindings_onto_mixin_attributes() -> None:
    """The structural-Protocol attributes (``mcp_servers``/``skills``/``rules``)
    must reflect the bindings the orchestrator granted; otherwise capability
    negotiation would see the mixin defaults instead of the live config."""
    binding = McpBinding(name="x", command=("/bin/mcp",), tools=("t",))
    caps = AllCapabilities(
        mcp_servers=(binding,),
        skills=SkillBinding(paths=("/sk",)),
        rules=AgentRules(text="rules"),
    )
    agent = ApiAgent(AgentConfig(capabilities=caps))
    assert agent.mcp_servers == (binding,)
    assert agent.skills == SkillBinding(paths=("/sk",))
    assert agent.rules == AgentRules(text="rules")


# ---------------------------------------------------------------------------
# Skills ⊥ MCP independence still holds under the binding-based config
# ---------------------------------------------------------------------------


def test_execute_runs_with_mcp_only_no_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    """MCP binding present, skills binding empty → MCP path opens, no skills loaded."""
    fc = [{"name": "do_thing", "args": {}, "id": "c1"}]
    fake = _FakeLLMClient([_Turn(text="working", calls=fc), _Turn(text="done")])
    mcp_tools = [SimpleNamespace(name="do_thing", description="d", inputSchema=None)]
    mcp = _FakeMCPClient(tools=mcp_tools)
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", lambda _path: mcp)

    result = ApiAgent(AgentConfig(capabilities=_mcp_caps("server"))).run("p")
    assert result.errors == []
    assert mcp.entered
    assert "skills_loaded" not in result.metadata  # SkillBinding stayed empty


def test_execute_runs_with_both_mcp_and_skills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both bindings populated → MCP session is opened *and* skills are discovered."""
    skill_dir = tmp_path / "skills"
    (skill_dir / "demo").mkdir(parents=True)
    (skill_dir / "demo" / "SKILL.md").write_text('---\nname: "demo"\ndescription: x\n---\nbody\n')

    fc = [{"name": "do_thing", "args": {}, "id": "c1"}]
    fake = _FakeLLMClient([_Turn(text="working", calls=fc), _Turn(text="done")])
    mcp = _FakeMCPClient(
        tools=[SimpleNamespace(name="do_thing", description="d", inputSchema=None)]
    )
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", lambda _path: mcp)

    caps = AllCapabilities(
        mcp_servers=(McpBinding(name="t", command=("server",)),),
        skills=SkillBinding(paths=(str(skill_dir),)),
    )
    result = ApiAgent(AgentConfig(capabilities=caps)).run("p")
    assert result.errors == []
    assert mcp.entered
    assert result.metadata["skills_loaded"] == ["demo"]


def test_execute_runs_with_neither_mcp_nor_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default capabilities → tool-less run, no MCP session, no skills loaded."""
    fake = _FakeLLMClient([_Turn(text="bare")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    # If MCPClient is ever entered the test would import mcp; replace with a sentinel.
    monkeypatch.setattr(
        agent_mod, "MCPClient", lambda _path: pytest.fail("MCPClient must not be used")
    )
    result = ApiAgent(AgentConfig()).run("p")
    assert result.output == "bare"
    assert result.errors == []
    assert "skills_loaded" not in result.metadata


# ---------------------------------------------------------------------------
# Rules flow into the loop's system_instruction
# ---------------------------------------------------------------------------


def test_execute_threads_rules_text_into_system_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty rules text must ride on the provider's ``system_instruction``."""
    fake = _FakeLLMClient([_Turn(text="ok")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    caps = AllCapabilities(rules=AgentRules(text="be careful"))
    ApiAgent(AgentConfig(capabilities=caps)).run("p")
    assert fake.calls[0]["system_instruction"] == "be careful"


def test_execute_empty_rules_text_yields_none_system_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty rules text → ``system_instruction=None`` (the loop's "no preamble")."""
    fake = _FakeLLMClient([_Turn(text="ok")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    ApiAgent(AgentConfig()).run("p")
    assert fake.calls[0]["system_instruction"] is None


# ---------------------------------------------------------------------------
# Mcp gate: an empty-command McpBinding does NOT open an MCP session in the
# API agent (it is for CLI agents whose binary launches MCP in-process).
# ---------------------------------------------------------------------------


def test_execute_skips_mcp_when_binding_has_no_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A binding with no launch command (CLI-agent shape) → API agent runs MCP-off."""
    fake = _FakeLLMClient([_Turn(text="ok")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(
        agent_mod, "MCPClient", lambda _path: pytest.fail("MCPClient must not be used")
    )
    caps = AllCapabilities(
        mcp_servers=(McpBinding(name="cli-shape", command=(), tools=("x",)),),
    )
    result = ApiAgent(AgentConfig(capabilities=caps)).run("p")
    assert result.output == "ok"


def test_execute_preserves_spaced_command_token_through_shlex_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spaced argv token (``("uv run", "mcp-server")``) must reach MCPClient
    intact: ``shlex.join`` quotes it on the way in, ``MCPClient``'s
    ``shlex.split`` recovers the original parts on the way out.

    Regression for the lossy ``" ".join`` that previously silently expanded
    one token into two when the binding carried a spaced word (e.g. a launch
    command that wraps an interpreter invocation).
    """
    import shlex

    captured: dict = {}

    class _RecordingMCP(_FakeMCPClient):
        def __init__(self, path: str) -> None:
            super().__init__(tools=[])
            captured["path"] = path

    fake = _FakeLLMClient([_Turn(text="done")])
    monkeypatch.setattr(agent_mod, "get_model", lambda *a, **kw: fake)
    monkeypatch.setattr(agent_mod, "MCPClient", _RecordingMCP)

    original = ("uv run", "mcp-server", "--flag")
    caps = AllCapabilities(
        mcp_servers=(McpBinding(name="spaced", command=original),),
    )
    ApiAgent(AgentConfig(capabilities=caps)).run("p")

    # MCPClient calls ``shlex.split`` on its ``server_path``; re-splitting the
    # path the agent handed it must recover the original tuple element-for-
    # element, including the spaced first token.
    assert tuple(shlex.split(captured["path"])) == original
