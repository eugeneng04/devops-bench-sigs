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

"""Check that the home fixtures a prompt promises actually reached the agent.

Five tasks seed an input next to the operator's home — a GitOps repo, a CVE
advisory, a rightsizing report — and point the prompt at ``~/<name>``. When
that file is not where the prompt says, the agent does not fail: it hunts the
filesystem, reconstructs what it can, and produces a score that looks ordinary.
Two whole batches were graded that way before anyone noticed, in two different
ways:

* seeded into a directory the agent's uid cannot read (fixtures written to
  ``/root`` at mode 0700 while the agent ran as uid 2000), and
* seeded relative to a different ``HOME`` than the agent's, so the path
  resolved somewhere the agent never looked.

Both are invisible from the score. This module turns them into a loud failure
before the agent starts, which is the only point where the distinction between
"the model could not do it" and "we never gave it the input" is still cheap to
make.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from devops_bench.core import get_bool, get_logger

__all__ = ["REQUIRE_FIXTURES_ENV", "check_prompt_fixtures", "prompt_fixture_paths"]

_log = get_logger("evalharness.fixtures")

#: Set to ``0`` to downgrade a missing fixture from a run-ending error to a
#: warning. For a deliberately degraded arm ("can the agent cope without its
#: source of truth?"), which is a different experiment from the graded task.
REQUIRE_FIXTURES_ENV = "BENCH_REQUIRE_FIXTURES"

# ``~/<name>`` or ``$HOME/<name>`` as it appears in prompt prose, usually inside
# quotes or backticks. The name runs to the first character that cannot be part
# of a filename in this corpus; trailing sentence punctuation is stripped below.
_HOME_PATH = re.compile(r"[~]/([\w.\-]+)|\$HOME/([\w.\-]+)")

# Stripped from the tail of a captured name: prose puts the path at the end of a
# sentence or clause far more often than a real fixture ends in punctuation.
_TRAILING_PUNCT = ".,;:!?'\"`)"


def prompt_fixture_paths(prompt: str, home: Path | None = None) -> list[Path]:
    """Extract the home-relative fixture paths a prompt promises the agent.

    The prompt is the contract: whatever it points the agent at with ``~/`` is
    something the stack was supposed to seed. Deriving the list from the prompt
    rather than from a per-task declaration keeps the two from drifting apart,
    and means a task that gains a fixture is covered the day its prompt
    mentions one.

    Args:
        prompt: The fully placeholder-substituted prompt text.
        home: Home directory to resolve against; defaults to the current
            process's, which is the agent's when it runs unsandboxed.

    Returns:
        Absolute paths, de-duplicated, in first-mention order.
    """
    base = home or Path.home()
    seen: dict[str, Path] = {}
    for match in _HOME_PATH.finditer(prompt):
        name = (match.group(1) or match.group(2) or "").rstrip(_TRAILING_PUNCT)
        if not name or name in seen:
            continue
        if prompt[match.end() : match.end() + 2] == "{{":
            # An unsubstituted ``{{CLUSTER_NAME}}`` follows, so the name is a
            # truncated prefix of the real one. Checking it would report a
            # missing fixture that was never named. Callers substitute before
            # checking; this only guards against being called out of order.
            continue
        seen[name] = base / name
    return list(seen.values())


def _unreadable_reason(path: Path) -> str | None:
    """Return why ``path`` is unusable as a fixture, or ``None`` if it is fine.

    Readability is checked with :func:`os.access`, which answers for *this*
    process. That is the right question when the agent runs as this user, and
    the closest available one otherwise — a sandboxed agent's view is covered
    by the mount plan instead.
    """
    if not path.exists():
        return "does not exist"
    if not os.access(path, os.R_OK):
        return "exists but is not readable by this user"
    if path.is_dir() and not os.access(path, os.X_OK):
        return "exists but is not traversable by this user"
    return None


def check_prompt_fixtures(
    prompt: str,
    task_name: str,
    home: Path | None = None,
    *,
    mounted: bool = False,
) -> list[str]:
    """Fail loudly when a promised fixture did not reach the agent.

    Args:
        prompt: The substituted prompt handed to the agent.
        task_name: Task name, for the message.
        home: Home directory the prompt's ``~`` resolves to for the agent.
        mounted: ``True`` when a sandbox mount plan already carried this run's
            fixtures into the container, in which case the host-side paths say
            nothing about what the agent can see and the check is skipped.

    Returns:
        One human-readable problem per unusable fixture; empty when every
        promised path is present and readable, or when there are none.

    Raises:
        RuntimeError: When a fixture is unusable and
            :data:`REQUIRE_FIXTURES_ENV` is not set to a false value.
    """
    if mounted:
        return []
    problems = [
        f"{path} ({reason})"
        for path in prompt_fixture_paths(prompt, home)
        if (reason := _unreadable_reason(path)) is not None
    ]
    if not problems:
        return []

    detail = "; ".join(problems)
    message = (
        f"task {task_name!r} promises the agent {len(problems)} home fixture(s) "
        f"it cannot read: {detail}. The stack did not seed them where this "
        f"agent looks, so the agent would hunt the filesystem and be graded on "
        f"whatever it could reconstruct. Fix the stack's seeding, or set "
        f"{REQUIRE_FIXTURES_ENV}=0 to run the task without its input on purpose."
    )
    if not get_bool(REQUIRE_FIXTURES_ENV, True):
        _log.warning("%s (continuing: %s is off)", message, REQUIRE_FIXTURES_ENV)
        return problems
    raise RuntimeError(message)
