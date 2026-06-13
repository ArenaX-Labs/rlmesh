from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rlmesh_system import runner
from rlmesh_system.support import manifest
from rlmesh_system.support.command import CommandError, run_command
from rlmesh_system.support.manifest import (
    EnvironmentSpec,
    ScenarioSpec,
    filter_scenarios,
    load_specs,
    select_environments,
)
from rlmesh_system.support.reports import compare_reports, regression_exit_code

PROFILES = Path(__file__).resolve().parents[3] / "tests" / "system" / "profiles"


def test_profiles_keep_system_surface_explicit() -> None:
    spec = load_specs(PROFILES)

    assert set(spec.profiles) == {
        "basic",
        "compatibility",
        "gymnasium",
        "heavy",
        "mujoco",
        "perf",
        "torch",
    }
    assert "stress" not in spec.profiles
    assert set(spec.environments) == {
        "basic-py310",
        "basic-py311",
        "gymnasium-py311",
        "mujoco-py311",
        "perf-jax-py311",
        "perf-numpy-py311",
        "perf-torch-py311",
        "torch-py311",
    }
    assert spec.scenarios["counter-entrypoint"].kind == "trace"
    assert spec.scenarios["tensor-numpy-view"].kind == "artifact"
    assert spec.profiles["compatibility"].environments == ()


def test_select_environments_defaults_to_basic() -> None:
    spec = load_specs(PROFILES)

    selected = select_environments([], [], spec)

    assert [environment.name for environment in selected] == [
        "basic-py310",
        "basic-py311",
    ]


def test_filter_scenarios_by_kind_and_name() -> None:
    spec = load_specs(PROFILES)
    environment = spec.environments["basic-py311"]

    selected = filter_scenarios(
        environment,
        spec,
        kinds={"trace"},
        scenario_names={"image-grid-numpy-action", "tensor-numpy-view"},
    )

    assert [scenario.name for scenario in selected] == ["image-grid-numpy-action"]


def test_platform_env_overrides_base_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "mujoco.toml").write_text(
        """
[profile]
environments = ["mujoco-py311"]

[environments.mujoco-py311]
python = "3.11"
scenarios = ["counter"]

[environments.mujoco-py311.env]
RLMESH_ENV = "base"
MUJOCO_GL = "base"

[environments.mujoco-py311.platform_env.linux]
MUJOCO_GL = "egl"

[environments.mujoco-py311.platform_env.darwin]
MUJOCO_GL = "cgl"

[scenarios.counter]
kind = "trace"
env = { fixture = "counter" }
model = "discrete.zero"
"""
    )

    monkeypatch.setattr(manifest.sys, "platform", "darwin")
    assert load_specs(profiles).environments["mujoco-py311"].env == {
        "RLMESH_ENV": "base",
        "MUJOCO_GL": "cgl",
    }

    monkeypatch.setattr(manifest.sys, "platform", "linux")
    assert load_specs(profiles).environments["mujoco-py311"].env == {
        "RLMESH_ENV": "base",
        "MUJOCO_GL": "egl",
    }


def test_env_server_command_supports_fixture_and_gym() -> None:
    spec = load_specs(PROFILES)
    fixture = spec.scenarios["counter-entrypoint"]
    gym = spec.scenarios["cartpole-trace"]

    fixture_command = runner.env_server_command(Path("/venv/bin/python"), fixture)
    gym_command = runner.env_server_command(Path("/venv/bin/python"), gym)

    assert "--entrypoint" in fixture_command
    assert "rlmesh_system_fixtures.registry:make_env" in fixture_command
    assert "--kwargs-json" in fixture_command
    assert any('"fixture": "counter"' in value for value in fixture_command)
    assert "--env" in gym_command
    assert "CartPole-v1" in gym_command
    assert "--package" in gym_command


def test_parse_server_address() -> None:
    assert (
        runner.parse_server_address("✓ Server address: tcp://127.0.0.1:54321")
        == "tcp://127.0.0.1:54321"
    )
    assert runner.parse_server_address("Waiting for client connection...") is None


def test_compare_warns_only_for_threshold_regressions(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    _write_report(
        baseline,
        [
            {
                "environment": "basic-py311",
                "name": "remote.small.step",
                "median_ms": 1.0,
                "p95_ms": 1.2,
            },
            {
                "environment": "basic-py311",
                "name": "tensor.numpy.asarray/1MiB",
                "median_ms": 0.10,
                "p95_ms": 0.12,
            },
        ],
    )
    _write_report(
        current,
        [
            {
                "environment": "basic-py311",
                "name": "remote.small.step",
                "median_ms": 1.4,
                "p95_ms": 1.5,
            },
            {
                "environment": "basic-py311",
                "name": "tensor.numpy.asarray/1MiB",
                "median_ms": 0.12,
                "p95_ms": 0.13,
            },
        ],
    )

    results = compare_reports(baseline, current)

    statuses = {(result.measurement, result.status) for result in results}
    assert ("remote.small.step", "regression") in statuses
    assert ("tensor.numpy.asarray/1MiB", "ok") in statuses
    assert regression_exit_code(results, fail_on_regression=False) == 0
    assert regression_exit_code(results, fail_on_regression=True) == 1


def test_run_command_times_out(tmp_path: Path) -> None:
    log_path = tmp_path / "timeout.log"

    with pytest.raises(CommandError) as exc_info:
        run_command(
            [
                sys.executable,
                "-c",
                "import time; time.sleep(10)",
            ],
            log_path=log_path,
            label="sleep",
            verbose=False,
            renderer=_Renderer(),
            timeout_seconds=0.2,
        )

    assert exc_info.value.returncode == 124
    assert "command timed out after 0.2s" in log_path.read_text()


def test_missing_external_command_fails_before_running(tmp_path: Path) -> None:
    scenario = ScenarioSpec(
        name="external-sim",
        kind="external",
        description="external",
        env={},
        model=None,
        client="numpy",
        seed=None,
        steps=1,
        trace=None,
        artifact=None,
        command_env="RLMESH_TEST_MISSING_COMMAND",
        timeout_seconds=None,
        metadata={},
    )
    environment = EnvironmentSpec(
        name="external-py311",
        python="3.11",
        dependencies=(),
        dependency_args=(),
        env={},
        tier="external",
        scenarios=("external-sim",),
        warmups=1,
        samples=1,
        processes=1,
        timeout_seconds=1,
        rlmesh={},
    )

    with pytest.raises(CommandError) as exc_info:
        runner.run_external_scenario(
            environment,
            scenario,
            {},
            tmp_path,
            verbose=False,
            renderer=_Renderer(),
        )

    assert exc_info.value.returncode == 2
    assert "requires RLMESH_TEST_MISSING_COMMAND" in exc_info.value.output


def test_dry_run_does_not_require_wheels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = runner.main(
        [
            "run",
            "--profile",
            "basic",
            "--wheel-dir",
            str(tmp_path / "missing-dist"),
            "--dry-run",
            "--plain",
        ]
    )

    assert code == 0
    output = capsys.readouterr().out
    assert "System validation dry run" in output
    assert "basic-py310" in output
    assert "basic-py311" in output
    assert "counter-entrypoint:trace" in output


def test_report_wheel_selection_warns_on_stale_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from datetime import datetime, timezone

    from rlmesh_system.support import wheel as wheel_module

    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    wheel = wheel_dir / "rlmesh-0.1.0-cp311-abi3-linux_x86_64.whl"
    wheel.write_bytes(b"")

    monkeypatch.setattr(
        wheel_module,
        "newest_source_commit_time",
        lambda _root: datetime(2100, 1, 1, tzinfo=timezone.utc),
    )

    environment = EnvironmentSpec(
        name="basic-py311",
        python="3.11",
        dependencies=(),
        dependency_args=(),
        env={},
        tier="basic",
        scenarios=(),
        warmups=1,
        samples=1,
        processes=1,
        timeout_seconds=1,
        rlmesh={},
    )

    runner.report_wheel_selection([environment], wheel_dir, runner.Renderer(plain=True))

    output = capsys.readouterr().out
    assert wheel.name in output
    assert "WARNING" in output


def _write_report(path: Path, measurements: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": 1, "measurements": measurements}))


class _Renderer:
    def step(self, label: str) -> None:
        pass

    def command(self, command: list[str]) -> None:
        pass

    def command_output(self, output: str) -> None:
        pass

    def command_running(self, label: str, elapsed: float, log_path: Path) -> None:
        pass
