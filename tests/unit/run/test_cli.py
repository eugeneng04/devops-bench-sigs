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

"""Tests for the thin argparse CLI (parser, config merge, exit codes)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from devops_bench.cli import args_to_config, build_parser, main
from devops_bench.core import ConfigError
from devops_bench.run import BenchmarkResult


@pytest.fixture(autouse=True)
def _restore_devops_bench_logger() -> Iterator[None]:
    """Undo ``main``'s ``configure_logging`` side effects (handlers, propagation,
    level), which otherwise break ``caplog`` in later tests."""
    import logging

    root = logging.getLogger("devops_bench")
    saved_handlers = list(root.handlers)
    saved_propagate = root.propagate
    saved_level = root.level
    yield
    root.handlers = saved_handlers
    root.propagate = saved_propagate
    root.setLevel(saved_level)


def test_build_parser_parses_flags() -> None:
    """Positional source and optional flags land on the namespace."""
    parser = build_parser()
    args = parser.parse_args(
        ["src", "--limit", "3", "--project", "p", "--cluster", "c", "--no-infra"]
    )
    assert args.source == "src"
    assert args.limit == 3
    assert args.project_id == "p"
    assert args.cluster_name == "c"
    assert args.no_infra is True
    # Tri-state: an unset flag is None (env fallback in args_to_config), not False.
    assert args.no_teardown is None


def test_infra_flag_forces_provisioning_over_truthy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--infra`` overrides a truthy ``BENCH_NO_INFRA`` back to False (tri-state pair)."""
    monkeypatch.setenv("BENCH_NO_INFRA", "true")
    parser = build_parser()
    args = parser.parse_args(["src", "--infra"])
    config = args_to_config(args)
    assert config.no_infra is False


def test_args_to_config_flags_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provided flags win over env; unset flags fall back to env."""
    monkeypatch.setenv("PROJECT_ID", "env-proj")
    monkeypatch.setenv("CLUSTER_NAME", "env-clus")
    monkeypatch.setenv("EVAL_LIMIT", "9")
    parser = build_parser()
    args = parser.parse_args(["src", "--project", "flag-proj"])
    config = args_to_config(args)
    # Flag override.
    assert config.project_id == "flag-proj"
    # Env fallbacks (no flag given).
    assert config.cluster_name == "env-clus"
    assert config.limit == 9


def test_args_to_config_bools_or_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Boolean flags OR the env value (env stays a fallback)."""
    monkeypatch.setenv("BENCH_NO_TEARDOWN", "true")
    monkeypatch.delenv("BENCH_NO_INFRA", raising=False)
    parser = build_parser()
    args = parser.parse_args(["src", "--no-infra"])
    config = args_to_config(args)
    assert config.no_infra is True  # from flag
    assert config.no_teardown is True  # from env


def _result(results: list[dict[str, object]], tmp_path: Path) -> BenchmarkResult:
    return BenchmarkResult(
        results=results,
        run_dir=tmp_path,
        results_path=tmp_path / "results.json",
        rows_path=tmp_path / "rows.json",
        manifest_path=tmp_path / "manifest.json",
    )


def test_main_exit_zero_on_no_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "devops_bench.run.run_benchmark",
        lambda config: _result([{"status": "success"}], tmp_path),
    )
    assert main(["src", "--no-infra"]) == 0
    assert "0 failed" in capsys.readouterr().out


def test_main_exit_one_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "devops_bench.run.run_benchmark",
        lambda config: _result([{"status": "success"}, {"status": "failed"}], tmp_path),
    )
    assert main(["src", "--no-infra"]) == 1


def test_main_exit_two_on_config_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _raise(config: object) -> BenchmarkResult:
        raise ConfigError("boom")

    monkeypatch.setattr("devops_bench.run.run_benchmark", _raise)
    assert main(["src"]) == 2
    assert "error: boom" in capsys.readouterr().err


def test_main_exit_two_on_missing_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A nonexistent source is a config error (exit 2); the loader normalizes it to ConfigError."""
    monkeypatch.delenv("BENCH_PARALLEL", raising=False)
    missing = tmp_path / "does-not-exist.yaml"
    assert main([str(missing), "--no-infra"]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "not found" in err


def test_main_exit_two_on_malformed_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A config error while building the config exits 2; ``args_to_config`` sits inside the try."""
    monkeypatch.setenv("EVAL_LIMIT", "abc")
    assert main(["src", "--no-infra"]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "EVAL_LIMIT" in err


def test_main_exit_two_on_unknown_log_level(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown ``BENCH_LOG_LEVEL`` is a configuration error like any other."""
    monkeypatch.setenv("BENCH_LOG_LEVEL", "LOUD")
    assert main(["src", "--no-infra"]) == 2
    err = capsys.readouterr().err
    assert "error:" in err
    assert "BENCH_LOG_LEVEL" in err
    assert "LOUD" in err


def test_help_works_even_with_an_unknown_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Logging is configured after parsing, so ``--help`` never depends on the env."""
    monkeypatch.setenv("BENCH_LOG_LEVEL", "LOUD")
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_main_attaches_a_real_log_handler(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The package logger's ``NullHandler`` silences every warning unless ``main`` configures logging."""
    import logging

    monkeypatch.setattr(
        "devops_bench.run.run_benchmark",
        lambda config: _result([{"status": "success"}], tmp_path),
    )
    main(["src", "--no-infra"])
    root = logging.getLogger("devops_bench")
    assert any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in root.handlers
    )
