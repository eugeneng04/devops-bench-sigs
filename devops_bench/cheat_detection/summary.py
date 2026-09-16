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

"""Turn a stored ``cheating_report`` into publishable one-line verdicts.

The detector owns the findings; this module owns the sentence each fired rule
contributes to a leaderboard row. Excerpts never cross that line — they hold
captured file content — so a verdict names the material and where it was
touched, and nothing else.

The wording splits on a finding's ``field``: ``args`` means the agent typed the
material into a tool call, ``result``/``output`` means it merely appeared in
front of the agent. Both trip the gate; they are not the same accusation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["describe_findings"]

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
# Locations named in one verdict before the rest become a count.
_MAX_LOCATIONS = 3
# Sorts last: a finding with no trajectory index came from the final output.
_LAST = float("inf")


def _ordered(pairs: list[Any]) -> list[Any]:
    """Sort ``(index, tool)`` pairs by trajectory position, indexless last."""
    return sorted(pairs, key=lambda w: (_LAST if w[0] is None else w[0], w[1]))


def _location(index: int | None, tool: str) -> str:
    if index is None:
        return "the run's final output"
    return f"{tool or 'a tool call'} at step {index}"


def _verdict(material: str, evidence: str, typed: list[Any], surfaced: list[Any]) -> str:
    """Compose one rule's sentence from where its findings landed.

    Typed access leads when present: the agent putting the material in a
    command is the stronger claim, and a sighting alongside it is a footnote
    rather than a second verdict.
    """
    if typed:
        shown = typed[:_MAX_LOCATIONS]
        line = f"{material}: read by the agent — " + ", ".join(_location(*w) for w in shown)
        extra = []
        if len(typed) > len(shown):
            extra.append(f"+{len(typed) - len(shown)} more reads")
        if surfaced:
            extra.append(f"+{len(surfaced)} passive sighting{'s' if len(surfaced) > 1 else ''}")
        return f"{line} ({', '.join(extra)})" if extra else line
    index, tool = surfaced[0]
    where = "the run's final output" if index is None else f"{tool or 'a tool call'} output"
    step = "" if index is None else f" at step {index}"
    line = f"{material}: {evidence} appeared in {where}{step}"
    if len(surfaced) > 1:
        line += f" (+{len(surfaced) - 1} more)"
    return line


def describe_findings(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return one ``{name, reason}`` verdict per rule that fired.

    Ordered by how much each implicates the run: typed access first, then
    severity, then trajectory position. Reports written before detector v8
    carry no rule ids, so they fall back to bare category names — the
    strongest statement their findings support.
    """
    groups: dict[str, dict[str, Any]] = {}
    findings = report.get("findings")
    for finding in findings if isinstance(findings, list) else []:
        if not isinstance(finding, Mapping):
            continue
        rule = finding.get("rule")
        if not isinstance(rule, str) or not rule:
            continue
        index = finding.get("trajectory_index")
        group = groups.setdefault(
            rule,
            {
                "material": finding.get("material") or rule,
                "evidence": finding.get("evidence") or "the path",
                "severity": finding.get("severity"),
                "typed": [],
                "surfaced": [],
            },
        )
        where = (index if isinstance(index, int) else None, finding.get("tool") or "")
        group["typed" if finding.get("field") == "args" else "surfaced"].append(where)

    if not groups:
        categories = report.get("categories")
        if not isinstance(categories, list):
            return []
        return [{"name": c, "reason": ""} for c in categories if isinstance(c, str)]

    def rank(item: tuple[str, dict[str, Any]]) -> tuple[int, int, float]:
        g = item[1]
        steps = [w[0] for w in g["typed"] + g["surfaced"] if w[0] is not None]
        return (
            0 if g["typed"] else 1,
            _SEVERITY_RANK.get(g["severity"], len(_SEVERITY_RANK)),
            min(steps, default=_LAST),
        )

    return [
        {
            "name": rule,
            "reason": _verdict(
                g["material"], g["evidence"], _ordered(g["typed"]), _ordered(g["surfaced"])
            ),
        }
        for rule, g in sorted(groups.items(), key=rank)
    ]
