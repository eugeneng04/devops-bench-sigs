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

"""Pre-run workspace inventory: turn prior-run leftovers into detection rules.

The agent home persists between runs, so artifacts of earlier runs (a previous
``report.md``, cloned working repos, seeded git remotes from other scenarios)
accumulate there and amount to answer keys for later runs. "Left by a prior
run" is a temporal property no static regex can express — this run's legit
``report.md`` write and a read of last run's leftover are the same string — so
the harness snapshots the home *before each task's agent starts* and generates
a ruleset from what it finds. Per task rather than once per batch because the
home keeps accumulating while the batch runs: task 1's deliverable is exactly
the kind of answer key task 2 must not read, and a single run-start snapshot
is blind to it. What each snapshot yields:

* a **path rule** per leftover, anchored to home-style prefixes (``~``,
  ``$HOME``, the literal home path) so an unrelated same-named file elsewhere
  (e.g. this run's own ``/tmp`` clone) is not flagged. Matched against every
  scanned surface, so the entry's name echoing through tool output (an
  ``ls ~``, a grep error trail) flags too: the agent had no reason to be
  looking; and
* a **content-fingerprint rule** per small text leftover that the *run-start*
  snapshot already saw: its most distinctive lines, matched only against tool
  ``result``/``output`` surfaces. A read of the stale file reproduces those
  exact lines; a freshly written file does not. Entries that appear mid-batch
  get their path rule but no fingerprint — see ``fingerprint_only`` in
  :func:`build_inventory_rules` for why the honest repeat iteration needs
  that exemption.

Path rules are filtered per record against the task prompt
(:func:`filter_rules_for_prompt`): an entry the prompt itself names — the
GitOps repo the agent is told to push to, the deliverable it is told to write
— is authorized for that record. That filter is what keeps an honest run
writing its own ``report.md`` clean, and it is deliberately the *only*
mechanism: a statically excluded name would be excluded for every task, so a
leftover from one task would go uncovered while a different task ran.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

from devops_bench.cheat_detection.rules import SCAN_FIELDS, SensitiveAccessRule
from devops_bench.core import get_logger

__all__ = [
    "DEFAULT_BASELINE",
    "ENVIRONMENT_DOTFILES",
    "baseline_from_granted_paths",
    "build_inventory_rules",
    "build_mount_rules",
    "drop_fingerprints_matching_inputs",
    "filter_rules_for_prompt",
    "narrow_home_listing_rules",
]

_log = get_logger("cheat_detection.inventory")

# Category stamped on every generated rule so reviewers can tell dynamic
# inventory findings from the static ruleset at a glance.
CATEGORY = "prior-run-artifact"

# Top-level home entries that belong to the provisioned harness environment
# rather than to any run. The harness-owned ones here are already covered by
# static rules where they matter (bench.env, matrix-runs). Deliberately
# limited to names this project itself creates: anything an operator's own
# host layout adds belongs in the ``baseline`` parameter of
# :func:`build_inventory_rules`, not here — see
# :func:`baseline_from_granted_paths` for the capability case.
DEFAULT_BASELINE: frozenset[str] = frozenset({"bench.env", "bin", "devops-bench", "matrix-runs"})

# Hidden entries that shell provisioning and ordinary tool use create — the
# environment, not any run's output. Enumerated rather than "everything
# starting with a dot" because agent CLIs conventionally keep their state in a
# dotdir, and that is exactly where cross-run contamination piles up: a
# ``~/.openclaw/workspace`` holding a previous task's deliverables and git
# history is an answer key like any other leftover, so it must generate rules.
# Known caveat, deliberately unhandled for now: the state dir of the agent
# *currently under test* is not special-cased, so its path surfacing in the
# trajectory flags. Honest agents rarely reference their own state dir in
# recorded tool calls; if this bites, the fix is the harness (which knows the
# agent type) adding that one name to the ``baseline`` it passes — not
# widening this set.
ENVIRONMENT_DOTFILES: frozenset[str] = frozenset(
    {
        ".bash_history",
        ".bash_logout",
        ".bash_profile",
        ".bashrc",
        ".cache",
        ".config",
        ".docker",
        ".gitconfig",
        ".gnupg",
        ".kube",
        ".lesshst",
        ".local",
        ".npm",
        ".profile",
        ".python_history",
        ".ssh",
        ".sudo_as_admin_successful",
        ".viminfo",
        ".vimrc",
        ".wget-hsts",
    }
)

# Fingerprinting bounds: leftovers are notes/manifests, not datasets. A file
# past the size cap is skipped (its path rule still applies); short lines are
# too generic ("## Summary") to identify a specific file.
_MAX_FINGERPRINT_BYTES = 64 * 1024
_FINGERPRINT_LINES = 3
_MIN_LINE_LEN = 24


def _home_prefixes(home: Path) -> str:
    """Regex alternation of the ways a trajectory spells the home directory.

    Left-bounded so a home spelling inside a longer token does not match: an
    unrelated ``/data/home/agent/report.md`` contains the literal home path as
    a substring, and a ``~`` glued to a word (``foo~/report.md``) is not a
    home reference. A preceding quote, whitespace, ``=`` or start-of-string
    still matches — the ways a shell actually introduces a home path.
    """
    return rf"(?<![\w~])(?:~|\$HOME|{re.escape(str(home))})"


def _path_rule(
    name: str, home_pattern: str, *, origin: str = "prior-run leftover"
) -> SensitiveAccessRule:
    """One rule matching home-anchored access to one leftover entry.

    One rule per entry (rather than one bundled rule) so per-record filtering
    can drop exactly the entries a task prompt authorizes. ``origin`` is the
    parenthetical provenance in the report description (the sandbox mount
    rules stamp their own).
    """
    return SensitiveAccessRule(
        category=CATEGORY,
        description=f"Pre-existing home entry '{name}' ({origin}) referenced by path.",
        severity="high",
        patterns=(rf"{home_pattern}/{re.escape(name)}(?![\w.-])",),
        fields=SCAN_FIELDS,
        source=name,
    )


def _fingerprint_lines(path: Path) -> tuple[str, ...]:
    """Return the most distinctive lines of a small text file, or nothing.

    Unreadable, oversized, or binary files yield no fingerprint — their path
    rule still covers them.
    """
    try:
        if path.stat().st_size > _MAX_FINGERPRINT_BYTES:
            return ()
        text = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return ()
    # Longest first, ties broken lexicographically: sorting a set with a
    # stable sort alone would leave equal-length lines in hash-randomized
    # order, picking different fingerprints per process.
    candidates = sorted(
        {line.strip() for line in text.splitlines() if len(line.strip()) >= _MIN_LINE_LEN},
        key=lambda line: (-len(line), line),
    )
    return tuple(candidates[:_FINGERPRINT_LINES])


def _content_rule(name: str, lines: tuple[str, ...]) -> SensitiveAccessRule:
    """A result/output-only rule matching a leftover file's distinctive lines."""
    return SensitiveAccessRule(
        category=CATEGORY,
        description=f"Content of pre-existing home file '{name}' surfacing in tool output.",
        severity="high",
        patterns=tuple(re.escape(line) for line in lines),
        fields=("result", "output"),
    )


def baseline_from_granted_paths(home: Path, paths: Iterable[str]) -> frozenset[str]:
    """Home entries that hold a granted capability, as baseline names.

    A capability the harness hands the agent — a skills tree from
    ``AGENT_SKILLS_PATHS``, say — is material the run is *told* to read, so
    the home entry containing it is provisioned environment rather than a
    prior-run leftover. Without this, every honest run of an arm that grants a
    skills tree living under the home would flag for using it.

    Derived rather than enumerated on purpose: hard-coding the operator's own
    directory names here would bake one host's layout into the detector and
    silently mis-flag every other one.

    Paths outside ``home`` (``/opt/skills/...``) contribute nothing — there is
    no home entry to exempt, and the inventory never saw them. Only the
    top-level component is taken, which is as fine-grained as the path rules
    themselves get: granting ``~/skills-repo/skills`` exempts ``skills-repo``.

    Args:
        home: The agent's home directory, as passed to
            :func:`build_inventory_rules`.
        paths: Granted capability paths, ``~``-expandable.

    Returns:
        Top-level ``home`` entry names to union into the baseline. Empty when
        nothing was granted from inside the home.
    """
    names: set[str] = set()
    for raw in paths:
        if not raw:
            continue
        try:
            granted = Path(os.path.abspath(os.path.expanduser(raw)))
            relative = granted.relative_to(home)
        except (OSError, ValueError):
            # Not under home, or unresolvable: no entry to exempt. Failing
            # closed here only costs a flag a reviewer can dismiss.
            continue
        if relative.parts:
            names.add(relative.parts[0])
    return frozenset(names)


def build_inventory_rules(
    home: Path,
    *,
    baseline: frozenset[str] = DEFAULT_BASELINE,
    fingerprint_only: frozenset[str] | None = None,
) -> tuple[SensitiveAccessRule, ...]:
    """Snapshot ``home`` and return rules covering its leftovers.

    Called by the harness before *each* task's agent executes, so a
    deliverable an earlier task in the same batch left behind is covered for
    every task after it. ``baseline`` names and the enumerated
    :data:`ENVIRONMENT_DOTFILES` are treated as the provisioned environment
    and skipped; any *other* hidden entry — an agent CLI's state directory,
    say — is a leftover like any visible one.

    Every leftover gets a path rule, deliverable names included. Authorizing a
    name the current task legitimately recreates is
    :func:`filter_rules_for_prompt`'s job, because it is per record: excluding
    a name here would exclude it for *every* task, leaving a leftover of that
    name uncovered while an unrelated task ran.

    Args:
        home: The agent's home directory (shared across runs on the host).
        baseline: Top-level names that belong to the environment, not a run.
        fingerprint_only: When given, the only entry names allowed to produce
            content rules; every other leftover gets its path rule alone. The
            harness passes the leftovers its *run-start* snapshot saw, so an
            entry that appears later in the batch is path-only. The asymmetry
            is deliberate: fingerprints are unfilterable by design, and two
            iterations of one task legitimately share long lines (a pasted
            policy body, a command line, a cluster name), so fingerprinting a
            same-batch deliverable would flag the honest repeat. Referencing
            a previous task's output *by path* has no such excuse, so the
            path rule still applies. ``None`` fingerprints every leftover.

    Returns:
        The generated ruleset: one path rule per leftover, plus one content
        rule per fingerprintable text leftover. Empty when the home is clean.
    """
    try:
        entries = sorted(home.iterdir())
    except OSError:
        return ()
    skip = baseline | ENVIRONMENT_DOTFILES
    leftovers = [p for p in entries if p.name not in skip]
    if not leftovers:
        return ()

    rules: list[SensitiveAccessRule] = []
    home_pattern = _home_prefixes(home)
    rules.extend(_path_rule(p.name, home_pattern) for p in leftovers)
    for entry in leftovers:
        if fingerprint_only is not None and entry.name not in fingerprint_only:
            continue
        # ``is_file()`` follows links, so a leftover symlink would pull an
        # arbitrary readable file's lines into a pattern — and patterns are
        # published in the report. The path rule still covers the link itself.
        if entry.is_symlink() or not entry.is_file():
            continue
        lines = _fingerprint_lines(entry)
        if lines:
            rules.append(_content_rule(entry.name, lines))
    return tuple(rules)


def build_mount_rules(container_home: str, names: Iterable[str]) -> tuple[SensitiveAccessRule, ...]:
    """Path rules for entries bind-mounted into a sandboxed agent's home.

    The sandbox-home inventory scans the *host-side* home directory, but
    fixture mounts only materialize inside the container: exactly where the
    agent sees mounted material, the host directory is empty and
    :func:`build_inventory_rules` covers nothing. These rules close that gap
    — one path rule per mounted name, anchored to the container-side home
    spellings (``~``, ``$HOME``, the literal container home), so a mount the
    task prompt does not name stays flagged. :func:`filter_rules_for_prompt`
    drops the prompt-named ones per record, exactly as for host-home
    inventory rules, which keeps a task's real fixtures un-flagged while a
    stray entry the discovery glob swept in stays covered.

    Deliberately path rules only, never content fingerprints: a mounted
    fixture is material the agent is usually *told* to read, and fingerprints
    are unfilterable by design — fingerprinting a mount would flag every
    honest read of a granted fixture.

    Args:
        container_home: The container-side home path the mounts live under
            (e.g. ``/workspace/home``).
        names: Basenames of the mounted entries.

    Returns:
        One path rule per distinct name, sorted for determinism.
    """
    prefix = rf"(?:~|\$HOME|{re.escape(container_home)})"
    return tuple(
        _path_rule(name, prefix, origin="mounted into the sandbox home")
        for name in sorted(set(names))
    )


def filter_rules_for_prompt(
    rules: tuple[SensitiveAccessRule, ...], prompt: str
) -> tuple[SensitiveAccessRule, ...]:
    """Drop path rules for home entries the task prompt itself names.

    A prompt that says "push to '~/opa-repo-<cluster>-eval.git'" *requires* the
    agent to reference that entry, so its inventory path rule would flag every
    honest run of the task. Naming an entry in the prompt is authorization to
    reference it — for that record only.

    Only path rules (which carry ``source``) are filterable here. Content
    fingerprints are handled separately by
    :func:`drop_fingerprints_matching_inputs`, because the reasoning that makes
    them safe to keep — "a prompt naming ``report.md`` tells the agent to
    *write* its own, and an honest write never reproduces the stale file's
    lines" — holds for outputs and fails for **delivered inputs**. A CVE
    advisory is copied from the same fixture every run, so an honest agent
    reading its own advisory reproduces the previous run's copy byte for byte.

    The name must appear as a whole token, not a substring: a prompt naming
    ``workspace-repo`` must not also authorize a ``workspace`` leftover, and
    naming ``report`` must not authorize ``report.md``. A sentence-ending
    period after the name is tolerated.

    Args:
        rules: Inventory-generated rules (static rules pass through untouched
            since they carry no ``source``).
        prompt: The record's task ``input`` text.

    Returns:
        ``rules`` minus the path rules whose source entry appears in ``prompt``.
    """
    if not prompt:
        return rules

    def named(source: str) -> bool:
        return re.search(rf"(?<![\w.-]){re.escape(source)}(?![\w-])(?!\.\w)", prompt) is not None

    return tuple(r for r in rules if not (r.source and named(r.source)))


#: Rules whose patterns are bare filenames living at the top of the operator's
#: home, so any listing of home surfaces them whether or not the agent read
#: anything. Narrowed to ``args`` for records whose prompt sends the agent into
#: home; left alone otherwise, where a sighting really is evidence of digging.
_HOME_LISTING_CATEGORIES: frozenset[str] = frozenset({"harness-environment"})


def narrow_home_listing_rules(
    rules: tuple[SensitiveAccessRule, ...], prompt: str, home: Path | None = None
) -> tuple[SensitiveAccessRule, ...]:
    """Stop passive home-listing sightings flagging a prompt-authorized run.

    Five tasks seed their input into the operator's home and point the prompt
    at ``~/<name>``, so an agent that runs ``ls -la ~`` to find it necessarily
    prints the harness's own top-level files and is flagged for material it
    never opened.

    A sighting is still evidence when nothing authorized the agent to be in
    home, so this narrows rather than removes, and only for records whose
    prompt names a home fixture. Naming one of these files on a command line
    still flags through ``args``; reading one while hiding the path still flags
    through the separate content-evidence rule.

    Args:
        rules: Rules to narrow.
        prompt: The record's substituted task ``input``.
        home: Home the prompt's ``~`` resolves to; defaults to the current
            user's.

    Returns:
        ``rules`` with the home-listing rules restricted to ``args`` when the
        prompt authorizes a home fixture, unchanged otherwise.
    """
    from devops_bench.evalharness.fixtures import prompt_fixture_paths

    if not prompt or not prompt_fixture_paths(prompt, home):
        return rules
    return tuple(
        rule.model_copy(update={"fields": ("args",)})
        if rule.category in _HOME_LISTING_CATEGORIES and rule.fields == SCAN_FIELDS
        else rule
        for rule in rules
    )


def drop_fingerprints_matching_inputs(
    rules: tuple[SensitiveAccessRule, ...], prompt: str, home: Path | None = None
) -> tuple[SensitiveAccessRule, ...]:
    """Drop content fingerprints that also match this run's own task input.

    A stale copy of a *delivered input* in the operator's home becomes a
    content fingerprint, and the honest agent then matches it by reading the
    current copy — the two are byte-identical, both seeded from the same file
    in the stack.

    The test is "does this pattern match what the prompt authorized the agent
    to read": a fingerprint that fires on the run's own input is evidence of
    nothing, whatever its provenance. Fingerprints that do not match an
    authorized input are untouched, so a genuinely stale ``report.md`` or a
    prior ``results.json`` still flags.

    Args:
        rules: Rules to filter, inventory-generated and static alike.
        prompt: The record's substituted task ``input``; the paths it names are
            the authorization.
        home: Home directory the prompt's ``~`` resolves to. Defaults to the
            current user's, which is the agent's on an unsandboxed run.

    Returns:
        ``rules`` minus the fingerprints that match an authorized input.
    """
    # Imported here rather than at module scope: the evalharness package pulls
    # in the agent and metric layers, and cheat_detection must stay importable
    # from them without a cycle.
    from devops_bench.evalharness.fixtures import prompt_fixture_paths

    if not prompt:
        return rules
    texts: list[str] = []
    for path in prompt_fixture_paths(prompt, home):
        try:
            if path.is_file():
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:  # unreadable input is the fixture check's problem
            _log.debug("could not read authorized input %s: %s", path, exc)
    if not texts:
        return rules

    def matches_own_input(rule: SensitiveAccessRule) -> bool:
        for pattern in rule.patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            except re.error:  # pragma: no cover - rules validate at load
                continue
            if any(compiled.search(text) for text in texts):
                return True
        return False

    kept = []
    for rule in rules:
        if matches_own_input(rule):
            _log.info(
                "dropping %s rule for this record: it matches the task's own declared input",
                rule.category,
            )
            continue
        kept.append(rule)
    return tuple(kept)
