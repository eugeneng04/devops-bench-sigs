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

"""Parser for the serialized ADK event stream.

An ADK run yields ``Event`` objects; the harness serializes each one and hands
the resulting list here. Working on plain mappings rather than SDK objects keeps
this module import-light (no ``google.adk`` dependency) and lets the tests
exercise the fold against recorded event dumps.

Each event carries a ``content.parts`` list. Three part shapes matter:

| Part shape          | Meaning                                              |
|---------------------|------------------------------------------------------|
| ``function_call``   | The model asked for a tool (``id``, ``name``, ``args``) |
| ``function_response`| The tool answered (``id``, ``name``, ``response``)   |
| ``text``            | Assistant prose                                      |

Calls and responses are correlated by the ``id`` ADK stamps on both sides, so a
call and its result fold into one :class:`~devops_bench.agents.result.ToolCall`
rather than two trajectory entries.

Three event-level fields outside ``content`` carry the run's telemetry:
``timestamp`` (epoch seconds, on every event), ``usage_metadata`` (one block per
LLM call), and ``model_version`` (the id the provider reported, on the events
the model authored).
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from devops_bench.agents.result import ToolCall, empty_tokens
from devops_bench.agents.shared.telemetry import ParsedRun, note_model
from devops_bench.agents.shared.timing import merged_span_sec, parse_event_time

__all__: list[str] = ["parse_event_stream"]

# ``usage_metadata`` field -> the accumulator slot it feeds. ADK passes the
# google-genai usage block through verbatim, so these are the genai names.
_USAGE_FIELDS: dict[str, str] = {
    "prompt_token_count": "prompt",
    "cached_content_token_count": "cached",
    "candidates_token_count": "output",
    "thoughts_token_count": "reasoning",
    "total_token_count": "total",
}


def _int_or_none(value: object) -> int | None:
    """Coerce to ``int``, rejecting ``bool`` (a JSON ``true`` is not a count)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parts(event: Mapping[str, Any]) -> list[Any]:
    """Return an event's ``content.parts`` list, or ``[]`` when it has none."""
    content = event.get("content")
    if not isinstance(content, Mapping):
        return []
    parts = content.get("parts")
    return parts if isinstance(parts, list) else []


def _is_user_content(event: Mapping[str, Any]) -> bool:
    """Report whether the event's content is authored by the user.

    Tool results ride on ``role="user"`` content too, so this only gates *text*
    accumulation — never the function-call fold.
    """
    content = event.get("content")
    return isinstance(content, Mapping) and content.get("role") == "user"


def _response_text(response: Any) -> tuple[str, bool]:
    """Render a tool response as trajectory text and report whether it failed.

    Three response shapes reach here:

    * an MCP tool result — ``{"content": [{"type": "text", ...}], "isError": ...}``,
      rendered as the concatenated text blocks so the trajectory stays readable
      instead of carrying the envelope,
    * ADK's error convention — a mapping carrying a truthy ``error`` key,
    * any other tool return value, rendered as sorted JSON.

    Args:
        response: The ``function_response.response`` payload.

    Returns:
        A ``(text, is_error)`` tuple.
    """
    if not isinstance(response, Mapping):
        return ("" if response is None else str(response)), False

    is_error = bool(response.get("isError")) or bool(response.get("error"))

    blocks = response.get("content")
    if isinstance(blocks, list):
        texts = [
            block["text"]
            for block in blocks
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        if texts:
            return "\n".join(texts), is_error

    return json.dumps(response, sort_keys=True, default=str), is_error


def _accumulate_usage(usage: Any, sums: dict[str, int], seen: set[str]) -> None:
    """Add one event's ``usage_metadata`` into the running per-bucket sums.

    ADK reports usage per LLM call, so a multi-turn tool loop emits one block
    per turn and the run total is their sum. A field absent from every block
    stays out of ``seen`` and is reported as ``None`` (unavailable) rather than
    a fabricated zero.
    """
    if not isinstance(usage, Mapping):
        return
    for field, slot in _USAGE_FIELDS.items():
        count = _int_or_none(usage.get(field))
        if count is None:
            continue
        sums[slot] = sums.get(slot, 0) + count
        seen.add(slot)


def _canonical_tokens(sums: Mapping[str, int], seen: set[str]) -> dict[str, int | None]:
    """Map the accumulated genai counts onto the canonical token buckets.

    genai reports ``prompt_token_count`` as the *full* prompt with the cached
    read as a subset of it, while the canonical ``input`` bucket excludes the
    cached portion — so ``input`` is the difference, clamped at ``0`` so an
    over-reported cache read can never produce a negative bucket. ``cache_write``
    has no genai counterpart and stays ``None``.
    """
    tokens = empty_tokens()
    if not seen:
        return tokens

    prompt = sums.get("prompt") if "prompt" in seen else None
    cached = sums.get("cached") if "cached" in seen else None
    inp = max(prompt - cached, 0) if prompt is not None and cached is not None else prompt

    tokens.update(
        input=inp,
        cached=cached,
        reasoning=sums.get("reasoning") if "reasoning" in seen else None,
        output=sums.get("output") if "output" in seen else None,
        total=sums.get("total") if "total" in seen else None,
    )
    return tokens


def parse_event_stream(events: Sequence[Any]) -> ParsedRun:
    """Fold a serialized ADK event stream into the canonical result shape.

    The parser is lenient by design — an unrecognized part shape is skipped
    rather than guessed at — but it never drops a signal silently: an event
    carrying ADK's ``error_code`` / ``error_message`` and a tool response
    matching no call both land on the returned ``errors`` list.

    Partial events (ADK's streaming chunks) and thought parts are excluded from
    the output text: the former would duplicate the text they are chunks of, and
    the latter is reasoning the judge should not grade as an answer.

    Args:
        events: Serialized ``Event`` mappings in the order ADK yielded them.

    Returns:
        A :class:`~devops_bench.agents.shared.telemetry.ParsedRun`. A call whose
        result never arrived stays ``status="called"`` with ``result=None``.
        ``tool_wait_sec`` pairs each ``function_call`` with the event bearing its
        ``function_response``; ``served_models`` reads ``model_version``, and
        ``model_turns`` counts the events carrying ``usage_metadata``.
    """
    output_parts: list[str] = []
    errors: list[str] = []
    trajectory: list[ToolCall] = []
    # Pending calls carry their own start time, keyed by ADK's correlation id
    # with a FIFO queue for the id-less calls some models emit. Each id maps to
    # a queue rather than one call:
    # distinct calls can legitimately reuse an id, so responses are matched in
    # emission order rather than the second call overwriting the first.
    pending_by_id: dict[str, list[tuple[ToolCall, float | None]]] = {}
    pending_unkeyed: deque[tuple[ToolCall, float | None]] = deque()
    sums: dict[str, int] = {}
    seen: set[str] = set()
    served_models: list[str] = []
    turns = 0
    spans: list[tuple[float, float]] = []

    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"event {index}: unexpected type {type(event).__name__}")
            continue

        code = event.get("error_code")
        message = event.get("error_message")
        if code or message:
            errors.append(
                f"event {index} reported {code or 'an error'}: {message or '<no detail>'}"
            )

        usage = event.get("usage_metadata")
        if isinstance(usage, Mapping):
            turns += 1
        _accumulate_usage(usage, sums, seen)
        note_model(served_models, event.get("model_version"))

        partial = bool(event.get("partial"))
        user_content = _is_user_content(event)
        event_time = parse_event_time(event.get("timestamp"))

        for part in _parts(event):
            if not isinstance(part, Mapping):
                continue

            call = part.get("function_call")
            if isinstance(call, Mapping):
                args = call.get("args")
                entry = ToolCall(
                    name=str(call.get("name") or ""),
                    args=dict(args) if isinstance(args, Mapping) else {},
                )
                trajectory.append(entry)
                call_id = call.get("id")
                if call_id is None:
                    pending_unkeyed.append((entry, event_time))
                else:
                    pending_by_id.setdefault(str(call_id), []).append((entry, event_time))
                continue

            response = part.get("function_response")
            if isinstance(response, Mapping):
                started = _fold_response(response, pending_by_id, pending_unkeyed, errors, index)
                if started is not None and event_time is not None:
                    spans.append((started, event_time))
                continue

            # ``thought`` marks a reasoning part: it is not the answer.
            text = part.get("text")
            if (
                isinstance(text, str)
                and text
                and not partial
                and not user_content
                and not part.get("thought")
            ):
                output_parts.append(text)

    return ParsedRun(
        output="".join(output_parts),
        trajectory=[entry.to_dict() for entry in trajectory],
        tokens=_canonical_tokens(sums, seen),
        errors=errors,
        tool_wait_sec=merged_span_sec(spans),
        served_models=served_models,
        # 0 turns means the stream carried no usage at all, not a run that
        # never called the model -- a stream exists because one was called.
        model_turns=turns or None,
    )


def _fold_response(
    response: Mapping[str, Any],
    pending_by_id: dict[str, list[tuple[ToolCall, float | None]]],
    pending_unkeyed: deque[tuple[ToolCall, float | None]],
    errors: list[str],
    index: int,
) -> float | None:
    """Attach one ``function_response`` to the call it answers.

    Matching is by ADK's correlation ``id``; a response with no id is paired
    with the oldest id-less call still awaiting a result, which is exact for a
    stream that answers calls in the order they were made. A response matching
    nothing is reported on ``errors`` rather than dropped.

    Returns:
        The matched call's start time, or ``None`` when the response matched no
        call or that call's event carried no usable timestamp -- either way the
        caller has nothing to time.
    """
    call_id = response.get("id")
    matched: tuple[ToolCall, float | None] | None = None
    if call_id is not None:
        queue = pending_by_id.get(str(call_id))
        matched = queue.pop(0) if queue else None
    elif pending_unkeyed:
        matched = pending_unkeyed.popleft()

    if matched is None:
        name = response.get("name") or "<unnamed>"
        errors.append(
            f"event {index}: tool response for {name!r} (id={call_id!r}) matched no pending call"
        )
        return None

    entry, started = matched
    entry.result, is_error = _response_text(response.get("response"))
    entry.status = "error" if is_error else "completed"
    return started
