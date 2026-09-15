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

"""Tests for the judged checklist metric.

What is pinned here is the line between "the agent failed" and "the judge could
not answer".
"""

from __future__ import annotations

from types import SimpleNamespace

from devops_bench.metrics import checklist as cl
from devops_bench.metrics.base import MetricScore

_EXPECTED = """critical requirements:
- alpha must hold
- beta must hold
- gamma must hold
"""


def _ctx() -> SimpleNamespace:
    """A context whose expected_output parses to three checklist items."""
    return SimpleNamespace(
        result={"expected_output": _EXPECTED},
        judge=None,
        use_mcp=False,
        outcome_case=None,
        tool_case=None,
        all_case=object(),
        generation_only=False,
    )


class _FakeGEval:
    """Stand-in for DeepEval's GEval: constructing the real one needs a key."""

    def __init__(self, name: str, **kwargs: object) -> None:
        self.name = name


def _stub_geval(monkeypatch) -> None:
    monkeypatch.setattr(cl, "GEval", _FakeGEval)


def test_checklist_is_withheld_when_the_judge_evaluates_nothing(monkeypatch) -> None:
    """Nothing judged means no opinion to publish — not a zero."""
    _stub_geval(monkeypatch)

    def dead_judge(case, metrics):
        raise RuntimeError("404 models/gemini-3.1-pro is not found")

    monkeypatch.setattr(cl, "run_geval", dead_judge)

    out = list(cl.ChecklistMetric().evaluate(_ctx()))

    assert out == [], "a dead judge must publish no ChecklistScore at all"


def test_checklist_scores_over_what_was_actually_judged(monkeypatch) -> None:
    """A partial outage shrinks the denominator; it does not fail the items."""
    _stub_geval(monkeypatch)
    calls = {"n": 0}

    def flaky(case, metrics):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("judge blew up on this one")
        return [MetricScore(name=f"Check: {calls['n']}", score=1.0, success=True)]

    monkeypatch.setattr(cl, "run_geval", flaky)

    out = list(cl.ChecklistMetric().evaluate(_ctx()))
    score = next(s for s in out if s.name == "ChecklistScore")

    assert score.score == 1.0, "both judged items passed, so the ratio is over 2 not 3"
    assert "could not be judged" in (score.reason or "")


def test_a_fully_judged_checklist_is_unchanged(monkeypatch) -> None:
    """The ordinary path keeps its previous meaning."""
    _stub_geval(monkeypatch)

    def judge(case, metrics):
        return [MetricScore(name="Check: x", score=0.0, success=False)]

    monkeypatch.setattr(cl, "run_geval", judge)

    out = list(cl.ChecklistMetric().evaluate(_ctx()))
    score = next(s for s in out if s.name == "ChecklistScore")

    assert score.score == 0.0
    assert "could not be judged" not in (score.reason or "")
    assert "0 out of 3" in (score.reason or "")
