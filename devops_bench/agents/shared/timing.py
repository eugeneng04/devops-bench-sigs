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

"""Turning agent-transcript timestamps into the wall clock a run spent in tools."""

from __future__ import annotations

import datetime

__all__ = ["merged_span_sec", "parse_event_time"]


def parse_event_time(value: object) -> float | None:
    """Parse a transcript event timestamp into epoch seconds.

    CLI transcripts stamp ISO-8601; the ADK event stream stamps epoch seconds
    already. A stamp with no offset is read as UTC so the result never depends
    on the runner's local zone. Missing or unparseable is ``None``, meaning "no
    timing available" rather than an error.
    """
    # ``bool`` first: ``True`` is an ``int``, and epoch second 1 is not a time.
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        stamped = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=datetime.UTC)
    return stamped.timestamp()


def merged_span_sec(intervals: list[tuple[float, float]]) -> float | None:
    """Return the wall-clock seconds covered by ``intervals``, overlaps counted once.

    A model turn can dispatch several tool calls that run concurrently, so
    summing their durations would report more tool time than the run took.
    ``intervals`` are ``(start, end)`` epoch-second pairs in any order; a pair
    ending before it starts is dropped as clock skew. ``None`` when none is
    usable — distinct from ``0.0``, which means the tools returned within the
    transcript's resolution.
    """
    usable = sorted((s, e) for s, e in intervals if e >= s)
    if not usable:
        return None
    total = 0.0
    cur_start, cur_end = usable[0]
    for start, end in usable[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + (cur_end - cur_start)
