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

"""Re-score stored runs whose grade was produced by a broken judge or detector.

Unlike :mod:`renormalize_rows`, which only re-derives the flat row contract from
scores already recorded, this **re-runs the metrics** — and for the judged ones
that means real model calls. It needs no cluster and no agent: every input it
uses (``output``, ``expected_output``, ``verification_report``, the trajectory)
is already in ``results.json``.

Two defects make a stored grade wrong rather than merely stale:

* **A judge that never answered.** With ``JUDGE_MODEL`` pointing at an id the
  endpoint does not serve, every GEval raised and each unjudged item was counted
  as a *failure*. The signature is unmistakable, and selecting on it is the
  default: a ``ChecklistScore`` reported beside **zero** ``Check:`` keys. Those
  verdicts do not exist anywhere, so they cannot be recomputed — only re-asked.
  Pass ``--all`` to re-judge every run regardless.
* **A detector false positive.** Merely listing the home directory, or reading
  the task's own delivered input, flagged runs that read nothing they should not
  have; the integrity gate then zeroed an otherwise good score. Re-scanning is
  local and free.

The original ``results.json`` is preserved before anything is written, in a
sibling tree mirroring the corpus layout (``--backup-root``), so the correction
stays auditable while the published tree keeps exactly its three artifacts per
run.

**Requires two fixes to be present in the tree**: the checklist withholding an
unjudged item instead of failing it, and the detector's prompt-authorization
filters. Without them this re-runs the same broken judge and the same rules,
and faithfully reproduces the numbers it was meant to correct — so ``--check``
verifies both before any work, and ``--write`` refuses without them.

Usage::

    uv run python scripts/rejudge_runs.py <root> --check     # inspect only
    uv run python scripts/rejudge_runs.py <root>             # dry run
    uv run python scripts/rejudge_runs.py <root> --write     # re-judge and save
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

#: Default sibling prefix holding each run's pre-re-judge record, mirroring the
#: corpus layout. A sibling rather than a file beside the run so the published
#: tree keeps exactly three artifacts per run and can be synced wholesale.
DEFAULT_BACKUP_DIRNAME = "rejudge-backups"


def _backup_path(run_dir: Path, backup_root: Path | None) -> Path:
    """Where this run's pre-re-judge record is preserved.

    Mirrors ``<combo>/<task>/<run_id>/results.json`` under the backup root, so
    a path in either tree names the same run.
    """
    root = backup_root or run_dir.parents[2].parent / DEFAULT_BACKUP_DIRNAME
    return root / run_dir.parents[1].name / run_dir.parent.name / run_dir.name / "results.json"


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _record(results: Any) -> dict[str, Any] | None:
    if isinstance(results, list):
        return results[0] if results and isinstance(results[0], dict) else None
    return results if isinstance(results, dict) else None


def needs_rejudge(record: dict[str, Any]) -> bool:
    """Whether this record carries the dead-judge signature.

    A judged checklist that reports a score while emitting no per-check verdict
    at all did not measure anything: every item raised and was counted as a
    failure. A task with no checklist (``ChecklistScore`` absent) is
    deterministic-only and is left alone.
    """
    scores = record.get("scores") or {}
    if scores.get("ChecklistScore") is None:
        return False
    return not any(k.startswith("Check:") for k in scores)


def was_flagged(record: dict[str, Any]) -> bool:
    """Whether detection flagged this record (so a re-scan may clear it)."""
    return (record.get("cheating_report") or {}).get("status") == "flagged"


def _fixes_present() -> tuple[bool, list[str]]:
    """Report whether the tree carries the fixes this tool depends on."""
    missing: list[str] = []
    try:
        from devops_bench.cheat_detection import (  # noqa: F401
            drop_fingerprints_matching_inputs,
            narrow_home_listing_rules,
        )
    except ImportError:
        missing.append("detector prompt-authorization filters")
    try:
        import inspect as _inspect

        from devops_bench.metrics import checklist as _cl

        if "could not be judged" not in _inspect.getsource(_cl.ChecklistMetric.evaluate):
            missing.append("checklist withholding on judge failure")
    except Exception:  # noqa: BLE001 - best effort probe
        missing.append("checklist withholding on judge failure (unreadable)")
    return (not missing), missing


def rejudge_run(
    run_dir: Path, judge: Any, *, write: bool, backup_root: Path | None = None
) -> dict[str, Any]:
    """Re-run detection and the metrics for one stored run.

    Args:
        run_dir: The ``<combo>/<task>/<run_id>`` directory to re-score.
        judge: The judge handed to the metrics pipeline.
        write: Whether to save; otherwise the metrics run and are discarded.
        backup_root: Where the pre-re-judge record is preserved. The mirror
            keeps the published tree to its three artifacts per run, so a
            consumer that syncs it does not also ship superseded scores.
    """
    from devops_bench.cheat_detection import (
        annotate_records,
        drop_fingerprints_matching_inputs,
        filter_rules_for_prompt,
        load_ruleset,
        narrow_home_listing_rules,
    )
    from devops_bench.metrics import evaluate_metrics_batch
    from devops_bench.results import Manifest, build_rows

    results = _load(run_dir / "results.json")
    record = _record(results)
    manifest_raw = _load(run_dir / "manifest.json") or {}
    before = dict(record.get("scores") or {}) if record else {}

    prompt = record.get("input") or ""
    # Inventory rules cannot be reconstructed after the fact — the home they
    # snapshotted has moved on — so the re-scan uses the static ruleset with the
    # same prompt-driven authorization the harness applies. A run flagged ONLY
    # by a stale inventory fingerprint therefore comes back clean, which is
    # precisely the false positive being corrected.
    rules = drop_fingerprints_matching_inputs(
        narrow_home_listing_rules(filter_rules_for_prompt(load_ruleset(), prompt), prompt),
        prompt,
    )
    annotate_records([record], rules)

    use_mcp = bool((record.get("capabilities_granted") or {}).get("use_mcp"))
    evaluate_metrics_batch([record], judge, use_mcp=use_mcp)

    after = record.get("scores") or {}
    summary = {
        "run": run_dir.name,
        "checks_before": sum(1 for k in before if k.startswith("Check:")),
        "checks_after": sum(1 for k in after if k.startswith("Check:")),
        "outcome_before": _score(before.get("OutcomeScore")),
        "outcome_after": _score(after.get("OutcomeScore")),
        "flagged_before": (record.get("cheating_report") or {}).get("status"),
    }

    if write:
        backup = _backup_path(run_dir, backup_root)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "results.json", backup)
        (run_dir / "results.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        # The manifest must name the judge that actually produced these scores.
        manifest_raw["judgeModel"] = getattr(judge, "_model_name", None) or getattr(
            judge, "model_name", None
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest_raw, indent=2) + "\n", encoding="utf-8"
        )
        fields = {
            "schema_version": manifest_raw.get("schemaVersion"),
            "run_id": manifest_raw.get("runId"),
            "t": manifest_raw.get("t"),
            "setup_id": manifest_raw.get("setupId"),
            "model": manifest_raw.get("model"),
            "harness": manifest_raw.get("harness"),
            "augmentation": manifest_raw.get("augmentation") or [],
            "judge_model": manifest_raw.get("judgeModel"),
        }
        rows = [r.to_dict() for r in build_rows([record], Manifest(**fields))]
        (run_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return summary


def _score(entry: Any) -> Any:
    return entry.get("score") if isinstance(entry, dict) else entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="corpus root: <combo>/<task>/<run_id>/")
    parser.add_argument("--write", action="store_true", help="save re-judged scores")
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help=f"where pre-re-judge records are preserved "
        f"(default: a sibling '{DEFAULT_BACKUP_DIRNAME}/' mirroring the corpus layout)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would be re-judged and whether the tree carries the fixes, then exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="re-judge every run, not only those with the dead-judge signature",
    )
    args = parser.parse_args(argv)

    ok, missing = _fixes_present()
    print("tree carries the required fixes: " + ("yes" if ok else f"NO — missing {missing}"))
    if not ok and args.write:
        print(
            "refusing to --write on a tree without the fixes: it would rewrite the "
            "same wrong numbers",
            file=sys.stderr,
        )
        return 2

    targets = []
    for results_path in sorted(args.root.glob("*/*/run_*/results.json")):
        record = _record(_load(results_path))
        if record is None:
            continue
        if args.all or needs_rejudge(record) or was_flagged(record):
            targets.append((results_path.parent, needs_rejudge(record), was_flagged(record)))

    print(f"\nruns selected: {len(targets)}")
    for run_dir, dead, flagged in targets:
        why = ", ".join(filter(None, ["dead judge" if dead else "", "flagged" if flagged else ""]))
        print(f"  {run_dir.parent.parent.name[:24]:26s} {run_dir.parent.name:22s} ({why})")
    if args.check or not targets:
        return 0

    from devops_bench.metrics import get_judge_model

    judge = get_judge_model()
    print(f"\njudge: {getattr(judge, '_model_name', judge)}")
    if not args.write:
        print("dry run: pass --write to save (metrics still run, results discarded)\n")

    for run_dir, _dead, _flagged in targets:
        s = rejudge_run(run_dir, judge, write=args.write, backup_root=args.backup_root)
        print(
            f"  {run_dir.parent.name:22s} checks {s['checks_before']}->{s['checks_after']}  "
            f"outcome {s['outcome_before']}->{s['outcome_after']}  "
            f"detector {s['flagged_before']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
