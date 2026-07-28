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

"""Shared fixtures for the deployer test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_ambient_bench_tf_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``BENCH_TF_ROOT`` so a developer's shell cannot skew resolution.

    The deployer resolves relative stacks against ``$BENCH_TF_ROOT`` when set;
    tests that exercise the override set it explicitly via ``monkeypatch``.
    """
    monkeypatch.delenv("BENCH_TF_ROOT", raising=False)
