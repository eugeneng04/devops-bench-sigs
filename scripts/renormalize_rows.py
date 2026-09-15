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

"""Rebuild ``rows.json`` from a run's stored ``results.json``.

Two defects in already-published runs are repairable from the artifacts alone,
without re-running an agent and without asking a judge anything:

* **Lost cache-token buckets.** The CLI harnesses emit ``cacheRead`` /
  ``cacheWrite`` / ``reasoningTokens``; the normalizer only read snake_case, so
  those buckets landed as ``null`` on every CLI row while ``total`` survived
  (spelled the same either way). The raw counts were always in ``results.json``.
* **A task folder of ``"task"``.** Older builds derived the folder from the
  spec file's stem, so ``<task-dir>/task.yaml`` became the literal ``"task"``
  for every task in the run. The dashboard groups tasks by ``taskFolder``, so a
  whole arm collapses into one cell — and because the bad value is also in
  ``results.json``, simply rebuilding the row reproduces it. The real folder is
  recoverable from the run's own directory layout.

This is deliberately **not** a rescore. It re-derives the flat row contract from
scores that are already recorded; it never re-evaluates a metric and never
constructs a judge. Rescoring would send every kept run back through a live
judge and move ``ChecklistScore`` underneath results that were reviewed as they
stand.

One consequence worth stating plainly: rebuilding a row applies the *current*
row rules to an *old* artifact. A run whose agent never finished loses its
published ``correctnessScore`` and ``outcomeScore``, because the current rules
refuse to derive either from a run the agent did not complete. That is a
correction, not a recomputation — the stored score is left untouched in
``results.json``, and only the row stops republishing it. Three runs in the
2026-09 corpus are affected, all already on the rerun list.

Layout expected (the published bucket's, and what the harness writes)::

    <root>/<combo>/<task>/<run_id>/{manifest,results,rows}.json

``<task>`` is the authority for the folder: it is the task directory basename
the run was launched from.

Usage::

    uv run python scripts/renormalize_rows.py <root>            # report only
    uv run python scripts/renormalize_rows.py <root> --write    # rewrite rows
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from devops_bench.results import Manifest, build_rows

#: The literal an older loader produced for every ``<task-dir>/task.yaml``.
_GENERIC_FOLDER = "task"


def _load_json(path: Path) -> Any:
    """Return parsed JSON, or ``None`` when the file is absent or malformed."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _records(results: Any) -> list[dict[str, Any]]:
    """Normalize ``results.json`` to a list of record dicts."""
    if isinstance(results, list):
        return [r for r in results if isinstance(r, dict)]
    return [results] if isinstance(results, dict) else []


def _manifest_from(raw: Any) -> Manifest | None:
    """Rebuild a :class:`Manifest` from stored ``manifest.json``.

    Unknown keys are dropped rather than raising: a manifest written by a build
    that carried extra fields must still round-trip through the current schema.
    """
    if not isinstance(raw, dict):
        return None
    fields = {
        "schema_version": raw.get("schemaVersion"),
        "run_id": raw.get("runId"),
        "t": raw.get("t"),
        "setup_id": raw.get("setupId"),
        "model": raw.get("model"),
        "harness": raw.get("harness"),
        "augmentation": raw.get("augmentation") or [],
        "judge_model": raw.get("judgeModel"),
    }
    try:
        return Manifest(**fields)
    except Exception:  # noqa: BLE001 - a malformed manifest skips its run, not the sweep
        return None


def _run_dirs(root: Path) -> Iterable[Path]:
    """Yield every run directory holding a ``results.json``, in sorted order."""
    return sorted(p.parent for p in root.glob("*/*/run_*/results.json"))


def renormalize_run(run_dir: Path, *, write: bool) -> dict[str, Any] | None:
    """Rebuild one run's rows, reporting what changed.

    Args:
        run_dir: A ``<combo>/<task>/<run_id>`` directory.
        write: When ``True``, overwrite ``rows.json``; otherwise report only.

    Returns:
        A summary mapping, or ``None`` when the run could not be read.
    """
    results = _load_json(run_dir / "results.json")
    manifest = _manifest_from(_load_json(run_dir / "manifest.json"))
    records = _records(results)
    if manifest is None or not records:
        return None

    # The directory name is the authority: it is the task directory the run was
    # launched from, and unlike the recorded folder it cannot have been
    # flattened to the spec file's stem.
    task_folder = run_dir.parent.name
    repaired_folder = 0
    for record in records:
        if str(record.get("folder") or "") in ("", _GENERIC_FOLDER):
            record["folder"] = task_folder
            repaired_folder += 1

    old_rows = _load_json(run_dir / "rows.json") or []
    new_rows = [row.to_dict() for row in build_rows(records, manifest)]

    def _cached(rows: Any) -> int:
        return sum(1 for r in rows if isinstance(r, dict) and r.get("cachedTokens") is not None)

    summary = {
        "run": str(run_dir),
        "rows": len(new_rows),
        "folder_repaired": repaired_folder,
        "cached_before": _cached(old_rows),
        "cached_after": _cached(new_rows),
        "changed": new_rows != old_rows,
    }
    if write and summary["changed"]:
        (run_dir / "rows.json").write_text(json.dumps(new_rows, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Sweep a corpus root, reporting (and optionally writing) rebuilt rows."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="corpus root holding <combo>/<task>/<run_id>/")
    parser.add_argument(
        "--write",
        action="store_true",
        help="overwrite rows.json; omit for a dry run that only reports",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    runs = list(_run_dirs(args.root))
    if not runs:
        print(f"error: no <combo>/<task>/<run_id>/results.json under {args.root}", file=sys.stderr)
        return 2

    unreadable = folder_fixed = cached_gained = changed = 0
    for run_dir in runs:
        summary = renormalize_run(run_dir, write=args.write)
        if summary is None:
            unreadable += 1
            print(f"  SKIP (unreadable manifest/results): {run_dir}")
            continue
        if summary["folder_repaired"]:
            folder_fixed += summary["folder_repaired"]
        gained = summary["cached_after"] - summary["cached_before"]
        cached_gained += max(0, gained)
        changed += int(bool(summary["changed"]))

    mode = "rewrote" if args.write else "would rewrite (dry run)"
    print(f"\nruns scanned      : {len(runs)}")
    print(f"unreadable        : {unreadable}")
    print(f"rows {mode:17s}: {changed}")
    # Reported as "overridden", not "repaired": the fix is written to rows.json
    # while results.json keeps its original folder, so every rebuild re-applies
    # the override and this count stays constant across runs. It is a property
    # of the source data, not a measure of work done this pass.
    print(f"taskFolder overridden from path: {folder_fixed} record(s)")
    print(f"cachedTokens recovered: {cached_gained} row(s)")
    if not args.write:
        print("\ndry run: pass --write to apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
