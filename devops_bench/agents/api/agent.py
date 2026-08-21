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

"""API/MCP agent harness driving the shared :func:`run_tool_loop` primitive.

:class:`ApiAgent` drives a model-agnostic MCP/skills tool-use loop and folds the
resulting conversation history into canonical :class:`ToolCall` trajectory
entries.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from devops_bench.agents.api.mcp import MCPClient, extract_tool_text
from devops_bench.agents.api.skills import (
    SkillToolInfo,
    discover_skill_tools,
)
from devops_bench.agents.base import AGENTS, AgentHarness
from devops_bench.agents.config import AgentConfig
from devops_bench.agents.result import AgentResult, ToolCall
from devops_bench.core import get_logger
from devops_bench.models import LLMClient, get_model
from devops_bench.models.utils.loop import LoopResult, ToolDispatcher, run_tool_loop

__all__ = ["ApiAgent", "fold_trajectory", "extract_tokens", "sum_tokens"]

_log = get_logger("agents.api.agent")

# Safety cap on agent turns; overridable via ``AgentConfig.max_turns``. Set high
# because API agents legitimately take many tool-use turns — it only guards
# against a model that never stops requesting tools.
_DEFAULT_MAX_TURNS = 50


def fold_trajectory(contents: list[dict]) -> list[dict]:
    """Fold a :class:`LoopResult.contents` history into canonical trajectory entries.

    Each assistant tool call is paired by ``tool_call_id`` with its tool result
    and emitted as one :class:`ToolCall`. Calls with no id — Gemini emits every
    function call with ``id=None`` — are paired FIFO with the id-less results,
    which is exact under :func:`run_tool_loop`'s contract of appending one
    result per dispatched call in call order.

    Args:
        contents: The conversation history produced by :func:`run_tool_loop`.

    Returns:
        A list of ``ToolCall.to_dict()`` mappings, one per tool call the model
        issued, in the order issued.

    Note:
        Tool results matching no assistant-issued call are dropped from the
        trajectory rather than synthesized into free-floating entries.
    """
    folded, _orphans = _fold_with_extraction_errors(contents)
    return folded


def _fold_with_extraction_errors(
    contents: list[dict],
) -> tuple[list[dict], list[str]]:
    """Same as :func:`fold_trajectory` but also returns orphan-result diagnostics.

    Args:
        contents: As :func:`fold_trajectory`.

    Returns:
        A ``(trajectory, orphan_errors)`` tuple. ``orphan_errors`` lists one
        message per unpaired ``role: tool`` entry; empty on a clean fold.
    """
    # Index assistant call ids first so we can detect orphans on the result
    # pass; id-less calls (Gemini emits every function call with ``id=None``)
    # are counted so their results can be paired FIFO instead of dropped.
    assistant_call_ids: set[str] = set()
    unkeyed_call_count = 0
    for msg in contents:
        if msg.get("role") != "assistant":
            continue
        for call in msg.get("tool_calls") or []:
            cid = call.get("id")
            if cid is None:
                unkeyed_call_count += 1
            else:
                assistant_call_ids.add(str(cid))

    # Pre-build call-id → (text, is_error) map so an out-of-order or absent
    # result still leaves the call as ``status="called"``/``result=None`` rather
    # than crashing on a lookup. Id-less results queue up FIFO — exact pairing
    # under run_tool_loop's append-one-result-per-call-in-order contract.
    results_by_id: dict[str, tuple[str, bool]] = {}
    unkeyed_results: deque[tuple[str, bool]] = deque()
    orphan_errors: list[str] = []
    for msg in contents:
        if msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        text = msg.get("content")
        text_str = text if isinstance(text, str) else "" if text is None else str(text)
        is_error = isinstance(text, str) and text.startswith("Error: ")
        if call_id is None and len(unkeyed_results) < unkeyed_call_count:
            unkeyed_results.append((text_str, is_error))
            continue
        if call_id is None or str(call_id) not in assistant_call_ids:
            # Drop the orphan from the trajectory but surface it for diagnostics.
            preview = text_str[:80].replace("\n", " ")
            msg_text = (
                "fold_trajectory dropped tool result with no matching call "
                f"(id={call_id!r}, content={preview!r})"
            )
            _log.debug(msg_text)
            orphan_errors.append(msg_text)
            continue
        results_by_id[str(call_id)] = (text_str, is_error)

    trajectory: list[ToolCall] = []
    for msg in contents:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args")
            call_id = call.get("id")
            entry = ToolCall(
                name=name,
                args=args if isinstance(args, dict) else {},
                status="called",
            )
            if call_id is not None:
                hit = results_by_id.get(str(call_id))
            else:
                hit = unkeyed_results.popleft() if unkeyed_results else None
            if hit is not None:
                text, is_error = hit
                entry.result = text
                entry.status = "error" if is_error else "completed"
            trajectory.append(entry)

    return [entry.to_dict() for entry in trajectory], orphan_errors


def extract_tokens(response: Any) -> dict[str, int | None]:
    """Map one turn's provider usage onto :data:`TOKEN_BUCKETS`.

    A tool-use loop bills every turn; use :func:`sum_tokens` for a whole run.

    ``input`` is the non-cached prompt and ``output`` excludes ``reasoning``,
    so the six buckets sum to ``total`` without double-counting. Whether a
    provider's prompt count already includes cached tokens differs per
    provider, so each shape is mapped explicitly rather than by alias:

    ============ ============================== ==================================
    Provider     Prompt count                   Buckets it cannot report
    ============ ============================== ==================================
    Google       includes cached — subtracted   ``cache_write``
    OpenAI       includes cached — subtracted   ``cache_write``
    Anthropic    excludes cache reads/writes    ``reasoning`` (billed in output)
    ============ ============================== ==================================

    An unreported ``cached`` / ``cache_write`` / ``reasoning`` bucket is
    ``None``, never ``0``, so the dashboard can tell "no cache telemetry" from
    "nothing was cached". ``input`` and ``output`` are always numbers — every
    provider reports both.

    Returns the buckets, or ``{}`` when the response carries no usage or a
    shape we do not recognize. ``{}`` reads downstream as "unavailable";
    zeros would claim a free run.
    """
    if response is None:
        return {}
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if usage is None:
        return {}

    # The three prompt-count names are mutually exclusive, so whichever is
    # present identifies the shape — and guarantees it to the mapper.
    if _int_attr(usage, "prompt_token_count") is not None:
        buckets = _google_buckets(usage)
    elif _int_attr(usage, "prompt_tokens") is not None:
        buckets = _openai_buckets(usage)
    elif _int_attr(usage, "input_tokens") is not None:
        buckets = _anthropic_buckets(usage)
    else:
        return {}

    # Anthropic reports no total at all, and Google zeroes an unset protobuf
    # int, so a 0 against non-zero buckets means "unset", not a free turn.
    if not buckets["total"]:
        buckets["total"] = sum(v for k, v in buckets.items() if k != "total" and v is not None)
    return buckets


def _google_buckets(usage: Any) -> dict[str, int | None]:
    """Map a Google ``usage_metadata`` object onto the canonical buckets.

    ``prompt_token_count`` includes cached tokens, so ``cached`` is subtracted
    back out. ``candidates_token_count`` excludes thinking, reported separately
    as ``thoughts_token_count``. Implicit caching carries no creation charge,
    so ``cache_write`` is unreported rather than zero.
    """
    prompt = _int_attr(usage, "prompt_token_count") or 0
    cached = _int_attr(usage, "cached_content_token_count")
    return {
        "input": max(prompt - (cached or 0), 0),
        "cached": cached,
        "cache_write": None,
        "reasoning": _int_attr(usage, "thoughts_token_count"),
        "output": _int_attr(usage, "candidates_token_count") or 0,
        "total": _int_attr(usage, "total_token_count"),
    }


def _openai_buckets(usage: Any) -> dict[str, int | None]:
    """Map an OpenAI / Ollama ``usage`` object onto the canonical buckets.

    Both split counts are *included* in their parent, so each is subtracted
    back out: ``prompt_tokens_details.cached_tokens`` sits inside
    ``prompt_tokens``, ``completion_tokens_details.reasoning_tokens`` inside
    ``completion_tokens``. Ollama omits both detail objects, leaving those
    buckets ``None``.

    OpenAI itself always satisfies ``total == prompt + completion``. A shim in
    front of a reasoning model need not: Gemini's bills thinking into
    ``total_tokens`` while leaving ``completion_tokens_details`` unset, so the
    thinking tokens land in no bucket. Any such shortfall is attributed to
    ``reasoning`` — the only bucket that can hold it, and the one the shim
    declined to fill.
    """
    prompt = _int_attr(usage, "prompt_tokens") or 0
    completion = _int_attr(usage, "completion_tokens") or 0
    cached = _nested_int_attr(usage, "prompt_tokens_details", "cached_tokens")
    reasoning = _nested_int_attr(usage, "completion_tokens_details", "reasoning_tokens")
    total = _int_attr(usage, "total_tokens")
    if reasoning is None and total is not None and total > prompt + completion:
        reasoning = total - prompt - completion
        output = completion
    else:
        output = max(completion - (reasoning or 0), 0)
    return {
        "input": max(prompt - (cached or 0), 0),
        "cached": cached,
        "cache_write": None,
        "reasoning": reasoning,
        "output": output,
        "total": total,
    }


def _anthropic_buckets(usage: Any) -> dict[str, int | None]:
    """Map an Anthropic ``usage`` object onto the canonical buckets.

    ``input_tokens`` already excludes cache reads and creation, so it maps
    untouched — the one provider here needing no subtraction. Thinking is
    billed inside ``output_tokens`` and not reported separately, so
    ``reasoning`` stays ``None``.
    """
    return {
        "input": _int_attr(usage, "input_tokens") or 0,
        "cached": _int_attr(usage, "cache_read_input_tokens"),
        "cache_write": _int_attr(usage, "cache_creation_input_tokens"),
        "reasoning": None,
        "output": _int_attr(usage, "output_tokens") or 0,
        "total": None,
    }


def sum_tokens(responses: Sequence[Any]) -> dict:
    """Sum :func:`extract_tokens` across every provider turn.

    An agentic loop re-sends the whole conversation each turn and is billed for
    each, so reading only the final response undercounts by close to the whole
    run — and the shortfall grows with turn count, the opposite of what a cost
    comparison needs.

    A bucket stays ``None`` unless some turn reported it; treating it as ``0``
    would turn "this provider sends no cache telemetry" into "nothing was
    cached". Returns ``{}`` when no turn reported usage.
    """
    totals: dict[str, int | None] = {}
    for response in responses:
        for key, value in extract_tokens(response).items():
            if value is None:
                totals.setdefault(key, None)
                continue
            running = totals.get(key)
            totals[key] = value if running is None else running + value
    return totals


def _int_attr(obj: Any, name: str) -> int | None:
    """Return ``obj.name`` as an int, or ``None`` when absent or unset.

    Provider usage objects are typed namespaces / pydantic models with the
    counts as plain ints; an unset field is either missing or ``None``. Both
    cases yield ``None`` so callers can tell an unreported bucket from a
    reported zero.
    """
    value = getattr(obj, name, None)
    return None if value is None else int(value)


def _nested_int_attr(obj: Any, parent: str, name: str) -> int | None:
    """Return ``obj.parent.name`` as an int, or ``None`` when either is absent.

    OpenAI nests its cached-prompt and reasoning counts one level down, and
    omits the whole detail object for models that report neither.
    """
    child = getattr(obj, parent, None)
    return None if child is None else _int_attr(child, name)


def _build_dispatch(
    mcp_client: MCPClient | None,
    skill_resources: dict[str, str],
    errors: list[str],
) -> ToolDispatcher:
    """Build the dispatcher passed to :func:`run_tool_loop`.

    Wraps each tool call in its own ``try/except`` because ``run_tool_loop``
    propagates dispatch errors by design — letting a single tool failure abort
    the whole run would be wrong for a benchmark agent. Failures are recorded
    on ``errors`` and returned as the tool result text so the model can react
    on the next turn.

    Args:
        mcp_client: Active :class:`MCPClient`, or ``None`` when MCP is off.
        skill_resources: Map of skill tool name to the skill file's content
            (captured at discovery); populated by
            :func:`devops_bench.agents.api.skills.discover_skill_tools`.
        errors: Errors list to mutate when a dispatch raises.

    Returns:
        An async ``(name, args, call_id) -> str`` callable matching
        :data:`devops_bench.models.utils.loop.ToolDispatcher`.
    """

    async def dispatch(name: str, args: Any, call_id: str | None) -> str:
        try:
            # Skill tools take priority — they are advertised in the same tool
            # list but are served locally from the content captured at
            # discovery, without round-tripping the MCP server or the disk.
            if name in skill_resources:
                _log.info("Serving skill tool %s from discovery-time content", name)
                return skill_resources[name]
            if mcp_client is None:
                msg = (
                    f"Error: tool {name!r} requested but no MCP server is "
                    "configured for this agent."
                )
                errors.append(msg)
                return msg
            arg_dict = args if isinstance(args, dict) else {}
            tool_result = await mcp_client.call_tool(name, arg_dict)
            text = extract_tool_text(tool_result)
            # MCP servers report tool failure via ``isError`` rather than
            # raising; surface it through the same error protocol as a raised
            # dispatch so the fold and metrics see the failure mode.
            if getattr(tool_result, "isError", False):
                msg = f"Error calling tool {name}: {text}"
                _log.warning(msg)
                errors.append(msg)
                return text if text.startswith("Error: ") else f"Error: {text}"
            return text
        except Exception as exc:  # noqa: BLE001 - one tool failure must not abort the run
            msg = f"Error calling tool {name}: {exc}"
            _log.warning(msg)
            errors.append(msg)
            return f"Error: {exc}"

    return dispatch


async def _gather_tools(
    mcp_client: MCPClient | None,
    skill_tools: list[SkillToolInfo],
) -> list[Any]:
    """Return the combined MCP + skill tool list passed to ``format_tools``.

    Args:
        mcp_client: Active MCP client, or ``None`` when MCP is off.
        skill_tools: Skill tool descriptors from
            :func:`discover_skill_tools`.

    Returns:
        A list of tool objects (MCP-native or :class:`SkillToolInfo`) in MCP
        order followed by skill order. Both are duck-typed (``name``,
        ``description``, ``inputSchema``) so adapters' ``format_tools`` handles
        them uniformly.
    """
    tools: list[Any] = []
    if mcp_client is not None:
        tools_result = await mcp_client.list_tools()
        tools.extend(tools_result.tools)
    tools.extend(skill_tools)
    return tools


async def _run_async(
    client: LLMClient,
    prompt: str,
    mcp_server_path: str | None,
    skills_paths: tuple[str, ...],
    rules_text: str | None,
    max_turns: int,
) -> tuple[LoopResult, list[str], list[str]]:
    """Drive the tool-use loop and return its ``(LoopResult, errors, skills)``.

    Opens an MCP session when ``mcp_server_path`` is set and discovers local
    skills when ``skills_paths`` is non-empty — the two are independent. The
    tool list is formatted by the caller and passed to :func:`run_tool_loop`
    pre-formatted.

    Args:
        client: Neutral LLM client.
        prompt: Task prompt seeding the loop.
        mcp_server_path: Command launching the MCP server, or ``None`` to skip
            MCP entirely.
        skills_paths: Filesystem locations to discover local skills under.
        rules_text: Operator-brief text (the ``AgentRules.text`` payload)
            handed to the provider as the ``system_instruction``; ``None`` /
            empty means "no preamble".
        max_turns: Safety cap on turns.

    Returns:
        A ``(loop_result, errors, skill_names)`` tuple. ``errors`` carries any
        per-tool dispatch failures recorded by the dispatcher.
    """
    errors: list[str] = []
    skill_tools, skill_resources, skill_names = await asyncio.to_thread(
        discover_skill_tools, skills_paths
    )
    system_instruction = rules_text or None

    if not mcp_server_path:
        formatted = client.format_tools(skill_tools)
        dispatch = _build_dispatch(None, skill_resources, errors)
        loop_result = await run_tool_loop(
            client=client,
            goal=prompt,
            tools=formatted,
            system_instruction=system_instruction,
            dispatch=dispatch,
            max_turns=max_turns,
        )
        return loop_result, errors, skill_names

    async with MCPClient(mcp_server_path) as mcp_client:
        tools = await _gather_tools(mcp_client, skill_tools)
        formatted = client.format_tools(tools)
        dispatch = _build_dispatch(mcp_client, skill_resources, errors)
        loop_result = await run_tool_loop(
            client=client,
            goal=prompt,
            tools=formatted,
            system_instruction=system_instruction,
            dispatch=dispatch,
            max_turns=max_turns,
        )
        return loop_result, errors, skill_names


@AGENTS.register("api")
class ApiAgent(AgentHarness):
    """API agent harness driving a model-agnostic MCP tool-use loop.

    Provider, model, and capability bindings all flow from
    :class:`~devops_bench.agents.config.AgentConfig` — no environment reads
    happen inside this class. Capability gates:

    * **MCP on/off** is driven by ``config.capabilities.mcp`` (presence of an
      :class:`~devops_bench.agents.capabilities.McpBinding` with a non-empty
      ``command``).
    * **Skills on/off** is driven by ``config.capabilities.skills.paths``
      independently of MCP — an agent may run with skills only, MCP only,
      both, or neither.
    * **Rules** flow from ``config.capabilities.rules.text`` and ride on the
      provider's ``system_instruction`` parameter (empty text → no preamble).

    ``__init__`` assigns ``self.mcp_servers`` / ``self.skills`` /
    ``self.rules`` from the config bindings, which makes
    ``isinstance(agent, SupportsMcp/SupportsSkills/SupportsRules)`` return
    ``True`` for orchestrator-side capability negotiation (the Protocols are
    structural — no mixin required).

    The execute path opens the MCP session (when configured), discovers
    skills, hands the *pre-formatted* tool list to :func:`run_tool_loop`, and
    folds the resulting conversation history into canonical
    :class:`~devops_bench.agents.result.ToolCall` trajectory entries via
    :func:`fold_trajectory`.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        # Assign the capability bindings as attributes so the structural
        # Protocols see the granted bindings.
        AgentHarness.__init__(self, config)
        caps = self.config.capabilities
        self.mcp_servers = caps.mcp_servers
        self.skills = caps.skills
        self.rules = caps.rules

    def _execute(self, prompt: str, workspace_path: Path | None = None) -> AgentResult:
        """Build the LLM client, drive the loop, and assemble an AgentResult.

        Args:
            prompt: Task prompt handed to the agent.
            workspace_path: Unused — the API agent drives a remote tool-use
                loop over MCP/skills with no local filesystem workspace.

        Returns:
            An :class:`AgentResult` whose ``trajectory`` is a list of canonical
            :class:`ToolCall` entries, ``output`` is :attr:`LoopResult.final_text`,
            and ``latency`` carries the loop's accumulated wall-clock.
            ``tokens`` is summed over every provider turn.
        """
        llm_client = get_model(self.config.provider, self.config.model)
        max_turns = (
            self.config.max_turns if self.config.max_turns is not None else _DEFAULT_MAX_TURNS
        )
        mcp_servers = self.config.capabilities.mcp_servers
        if len(mcp_servers) > 1:
            _log.warning(
                "API agent honors only the first MCP binding; ignoring %d additional server(s)",
                len(mcp_servers) - 1,
            )
        mcp_binding = self.config.capabilities.mcp
        # Only open an MCPClient when an MCP binding carries a launch command;
        # an empty-command binding is treated as "no MCP". ``shlex.join``
        # round-trips ``MCPClient``'s ``shlex.split`` so a spaced argv token
        # (``("uv run", "mcp-server")``) is rebuilt as a single quoted word.
        mcp_server_path = (
            shlex.join(mcp_binding.command) if mcp_binding and mcp_binding.command else None
        )
        skills_paths = self.config.capabilities.skills.paths
        rules_text = self.config.capabilities.rules.text

        run_coro = _run_async(
            llm_client,
            prompt,
            mcp_server_path,
            skills_paths,
            rules_text,
            max_turns,
        )
        # Honor the config's wall-clock budget over the whole run (MCP session
        # + every provider turn), matching the CLI harnesses' whole-call
        # timeout semantics. ``wait_for`` with ``timeout=None`` imposes none.
        timeout = self.config.timeout_sec
        start = time.monotonic()
        try:
            loop_result, dispatch_errors, skill_names = asyncio.run(
                asyncio.wait_for(run_coro, timeout=timeout)
            )
        except TimeoutError:
            # ``wait_for``'s TimeoutError can only fire once the deadline has
            # elapsed (and never with ``timeout=None``) — anything earlier came
            # from inside the run (e.g. a provider socket timeout;
            # socket.timeout is TimeoutError since 3.10). Re-raise those so the
            # base safety net logs the traceback instead of relabeling them.
            elapsed = time.monotonic() - start
            if timeout is None or elapsed < timeout:
                raise
            return AgentResult.errored(f"API agent timed out after {timeout}s", latency=elapsed)

        trajectory, orphan_errors = _fold_with_extraction_errors(loop_result.contents)
        tokens = sum_tokens(loop_result.responses)
        turns = [
            {
                "tokens": extract_tokens(turn.response),
                "latency_sec": turn.latency,
                "tool_calls": turn.tool_calls,
            }
            for turn in loop_result.turns
        ]
        metadata: dict[str, Any] = {
            "tools_used": sorted(loop_result.tools_used),
        }
        if skill_names:
            metadata["skills_loaded"] = list(skill_names)

        return AgentResult(
            output=loop_result.final_text,
            trajectory=trajectory,
            tokens=tokens,
            latency=loop_result.latency,
            turns=turns,
            errors=list(dispatch_errors) + orphan_errors,
            metadata=metadata,
        )
