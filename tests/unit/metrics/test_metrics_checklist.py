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

import pytest

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


def _stub_geval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cl, "GEval", _FakeGEval)


def _by_name(out: list[MetricScore]) -> dict[str, MetricScore]:
    return {ms.name: ms for ms in out}


def test_checklist_abstains_when_the_judge_evaluates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing judged means a null aggregate and a null entry per item — not a zero, not silence."""
    _stub_geval(monkeypatch)

    def dead_judge(case: object, metrics: list[_FakeGEval]) -> list[MetricScore]:
        raise RuntimeError("404 models/gemini-3.1-pro is not found")

    monkeypatch.setattr(cl, "run_geval", dead_judge)

    scores = _by_name(list(cl.ChecklistMetric().evaluate(_ctx())))

    aggregate = scores["ChecklistScore"]
    assert aggregate.score is None and aggregate.success is None
    assert aggregate.reason == "Withheld: 3 of 3 checks could not be judged."
    items = [ms for name, ms in scores.items() if name.startswith("Check: ")]
    assert len(items) == 3
    assert all(ms.score is None and "404" in (ms.reason or "") for ms in items)


def test_checklist_is_withheld_when_any_item_could_not_be_judged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One unjudged item withholds the aggregate; 2 of 2 judged is never published as 1.0."""
    _stub_geval(monkeypatch)
    calls = {"n": 0}

    def flaky(case: object, metrics: list[_FakeGEval]) -> list[MetricScore]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("judge blew up on this one")
        return [MetricScore(name=metrics[0].name, score=1.0, success=True)]

    monkeypatch.setattr(cl, "run_geval", flaky)

    scores = _by_name(list(cl.ChecklistMetric().evaluate(_ctx())))

    aggregate = scores["ChecklistScore"]
    assert aggregate.score is None and aggregate.success is None
    assert aggregate.reason == "Withheld: 1 of 3 checks could not be judged (2 of 2 judged passed)."
    skipped = scores["Check: beta must hold"]
    assert skipped.score is None and skipped.success is None
    assert "judge blew up" in (skipped.reason or "")


def test_a_fully_judged_checklist_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary path keeps its previous meaning."""
    _stub_geval(monkeypatch)

    def judge(case: object, metrics: list[_FakeGEval]) -> list[MetricScore]:
        return [MetricScore(name=metrics[0].name, score=0.0, success=False)]

    monkeypatch.setattr(cl, "run_geval", judge)

    scores = _by_name(list(cl.ChecklistMetric().evaluate(_ctx())))

    assert scores["ChecklistScore"].score == 0.0
    assert scores["ChecklistScore"].reason == "Passed 0 out of 3 checks."
    assert all(ms.score == 0.0 for name, ms in scores.items() if name.startswith("Check: "))


def test_an_empty_checklist_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No bullets means no aggregate, not a division by zero."""
    _stub_geval(monkeypatch)
    ctx = _ctx()
    ctx.result = {"expected_output": "no bullets here"}
    assert list(cl.ChecklistMetric().evaluate(ctx)) == []


def test_a_bare_error_keeps_its_type_in_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-task timeout re-raises with an empty message; the type is all there is."""
    _stub_geval(monkeypatch)

    def _run(case: object, metrics: list[_FakeGEval]) -> list[MetricScore]:
        raise TimeoutError

    monkeypatch.setattr(cl, "run_geval", _run)
    scores = _by_name(list(cl.ChecklistMetric().evaluate(_ctx())))
    assert scores["Check: alpha must hold"].reason == "Not judged: TimeoutError"


def test_an_item_with_no_verdict_is_unjudged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A judge that returns nothing for an item has not judged it."""
    _stub_geval(monkeypatch)

    def _run(case: object, metrics: list[_FakeGEval]) -> list[MetricScore]:
        if metrics[0].name.endswith("beta must hold"):
            return []
        return [MetricScore(name=metrics[0].name, score=1.0, success=True)]

    monkeypatch.setattr(cl, "run_geval", _run)
    aggregate = _by_name(list(cl.ChecklistMetric().evaluate(_ctx())))["ChecklistScore"]
    assert aggregate.score is None
    assert aggregate.reason == "Withheld: 1 of 3 checks could not be judged (2 of 2 judged passed)."
