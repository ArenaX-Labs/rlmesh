#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from rlmesh_system.support.command import CommandError, run_command
from rlmesh_system.support.env import prepare_installed_wheel_environment
from rlmesh_system.support.manifest import (
    EnvironmentSpec,
    ScenarioSpec,
    filter_scenarios,
    load_specs,
    profile_names_or_default,
    select_environments,
)
from rlmesh_system.support.process import start_process_until
from rlmesh_system.support.rendering import Renderer
from rlmesh_system.support.reports import (
    EnvironmentResult,
    aggregate_report,
    compare_reports,
    regression_exit_code,
)
from rlmesh_system.support.wheel import inspect_wheel

ROOT = Path(__file__).resolve().parents[4]
SYSTEM_DIR = ROOT / "tests" / "system"
DEFAULT_PROFILE_DIR = SYSTEM_DIR / "profiles"
DEFAULT_FIXTURE_DIR = ROOT / "tools" / "rlmesh_system_fixtures"
DEFAULT_TRACE_DIR = SYSTEM_DIR / "traces"
DEFAULT_WHEEL_DIR = ROOT / "python" / "rlmesh" / "dist"
DEFAULT_WORK_DIR = ROOT / "target" / "python-validation"
SERVER_ADDRESS_RE = re.compile(r"Server address:\s*(\S+)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run RLMesh installed-artifact system test profiles."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list profiles and scenarios")
    list_parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    list_parser.add_argument("--plain", action="store_true", help="disable Rich output")

    run_parser = subparsers.add_parser("run", help="run system scenarios")
    run_parser.add_argument("--profile", action="append", default=[])
    run_parser.add_argument("--environment", action="append", default=[])
    run_parser.add_argument("--scenario", action="append", default=[])
    run_parser.add_argument(
        "--kind",
        action="append",
        choices=["trace", "artifact", "external"],
        default=[],
        help="restrict selected environments to one or more scenario kinds",
    )
    run_parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    run_parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    run_parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    run_parser.add_argument("--wheel-dir", type=Path, default=DEFAULT_WHEEL_DIR)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print selected environments and scenarios without building venvs",
    )
    run_parser.add_argument("--baseline", type=Path)
    run_parser.add_argument("--fail-on-regression", action="store_true")
    run_parser.add_argument("--keep", action="store_true", help="keep existing venvs")
    run_parser.add_argument(
        "--verbose", action="store_true", help="print command output"
    )
    run_parser.add_argument("--plain", action="store_true", help="disable Rich output")

    compare_parser = subparsers.add_parser(
        "compare", help="compare two system measurement reports"
    )
    compare_parser.add_argument("baseline", type=Path)
    compare_parser.add_argument("current", type=Path)
    compare_parser.add_argument("--fail-on-regression", action="store_true")
    compare_parser.add_argument(
        "--plain", action="store_true", help="disable Rich output"
    )

    clean_parser = subparsers.add_parser(
        "clean", help="remove system validation venvs, logs, and reports"
    )
    clean_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    clean_parser.add_argument(
        "--plain", action="store_true", help="disable Rich output"
    )

    args = parser.parse_args(argv)
    renderer = Renderer(plain=getattr(args, "plain", False))

    if args.command == "list":
        spec = load_specs(args.profile_dir)
        renderer.print_profiles(spec.profiles, spec.environments, spec.scenarios)
        return 0
    if args.command == "run":
        spec = load_specs(args.profile_dir)
        selected = select_environments(args.profile, args.environment, spec)
        kinds = set(args.kind) if args.kind else None
        scenario_names = set(args.scenario) if args.scenario else None
        if args.dry_run:
            print_dry_run(
                selected,
                spec,
                args.wheel_dir,
                args.fixture_dir,
                args.work_dir,
                profile_names=profile_names_or_default(args.profile),
                kinds=kinds,
                scenario_names=scenario_names,
                renderer=renderer,
            )
            return 0
        return run_environments(
            selected,
            spec,
            args.wheel_dir,
            args.fixture_dir,
            args.trace_dir,
            args.work_dir,
            profile_names=profile_names_or_default(args.profile),
            kinds=kinds,
            scenario_names=scenario_names,
            baseline=args.baseline,
            fail_on_regression=args.fail_on_regression,
            keep=args.keep,
            verbose=args.verbose,
            renderer=renderer,
        )
    if args.command == "compare":
        results = compare_reports(args.baseline, args.current)
        renderer.regression_summary(results)
        return regression_exit_code(results, fail_on_regression=args.fail_on_regression)
    if args.command == "clean":
        shutil.rmtree(args.work_dir, ignore_errors=True)
        renderer.print(f"removed {args.work_dir}")
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")


def report_wheel_selection(
    environments: list[EnvironmentSpec],
    wheel_dir: Path,
    renderer: Renderer,
) -> None:
    """Print the wheel each Python version will install and warn if it is stale.

    ``uv pip install --find-links`` silently installs whatever matching wheel
    sits in ``wheel_dir``; a leftover pre-build wheel produces misleading
    numbers. Surface the selection (and a loud warning when it predates the
    sources) so a stale wheel never goes unnoticed.
    """
    seen: set[str] = set()
    for environment in environments:
        if environment.python in seen:
            continue
        seen.add(environment.python)
        status = inspect_wheel(wheel_dir, environment.python, root=ROOT)
        if status is None:
            renderer.print(
                f"python {environment.python}: no matching rlmesh wheel in {wheel_dir}"
            )
            continue
        renderer.print(f"python {environment.python}:")
        for line in status.header_lines():
            highlight = line.startswith("WARNING") and renderer.rich
            renderer.print(f"  [bold red]{line}[/]" if highlight else f"  {line}")
    renderer.print()


def print_dry_run(
    environments: list[EnvironmentSpec],
    spec: object,
    wheel_dir: Path,
    fixture_dir: Path,
    work_dir: Path,
    *,
    profile_names: list[str],
    kinds: set[str] | None,
    scenario_names: set[str] | None,
    renderer: Renderer,
) -> None:
    renderer.print("System validation dry run")
    renderer.print(f"profiles: {', '.join(profile_names)}")
    renderer.print(f"wheel_dir: {wheel_dir.resolve()}")
    renderer.print(f"fixture_dir: {fixture_dir.resolve()}")
    renderer.print(f"work_dir: {work_dir.resolve()}")
    if kinds:
        renderer.print(f"kinds: {', '.join(sorted(kinds))}")
    if scenario_names:
        renderer.print(f"scenarios: {', '.join(sorted(scenario_names))}")
    renderer.print()

    for environment in environments:
        scenarios = filter_scenarios(
            environment,
            spec,  # type: ignore[arg-type]
            kinds=kinds,
            scenario_names=scenario_names,
        )
        if not scenarios:
            renderer.print(f"{environment.name}: skipped (no scenarios selected)")
            continue

        dependencies = ", ".join(environment.dependencies) or "<none>"
        scenario_text = ", ".join(
            f"{scenario.name}:{scenario.kind}" for scenario in scenarios
        )
        renderer.print(
            f"{environment.name}: tier={environment.tier}, "
            f"python={environment.python}, dependencies={dependencies}"
        )
        renderer.print(f"  scenarios: {scenario_text}")


def run_environments(
    environments: list[EnvironmentSpec],
    spec: object,
    wheel_dir: Path,
    fixture_dir: Path,
    trace_dir: Path,
    work_dir: Path,
    *,
    profile_names: list[str],
    kinds: set[str] | None,
    scenario_names: set[str] | None,
    baseline: Path | None,
    fail_on_regression: bool,
    keep: bool,
    verbose: bool,
    renderer: Renderer,
) -> int:
    wheel_dir = wheel_dir.resolve()
    if environments and not list(wheel_dir.glob("rlmesh-*.whl")):
        raise SystemExit(
            f"no rlmesh wheels found in {wheel_dir}; run build:python:local first"
        )

    report_wheel_selection(environments, wheel_dir, renderer)

    work_dir.mkdir(parents=True, exist_ok=True)
    report_dir = work_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    results: list[EnvironmentResult] = []
    environment_reports: list[dict[str, Any]] = []
    for environment in environments:
        scenarios = filter_scenarios(
            environment,
            spec,  # type: ignore[arg-type]
            kinds=kinds,
            scenario_names=scenario_names,
        )
        if not scenarios:
            renderer.environment_skip(environment, "no scenarios selected")
            continue

        renderer.environment_start(environment, scenarios)
        started = time.monotonic()
        try:
            report = run_environment(
                environment,
                scenarios,
                wheel_dir,
                fixture_dir,
                trace_dir,
                work_dir,
                report_dir=report_dir,
                run_id=run_id,
                keep=keep,
                verbose=verbose,
                renderer=renderer,
            )
        except CommandError as exc:
            elapsed = time.monotonic() - started
            results.append(
                EnvironmentResult(
                    name=environment.name,
                    python=environment.python,
                    ok=False,
                    elapsed=elapsed,
                    log_path=exc.log_path,
                    returncode=exc.returncode,
                )
            )
            renderer.environment_failed(environment, elapsed, exc)
        else:
            elapsed = time.monotonic() - started
            report_path = Path(str(report["report_path"]))
            environment_reports.append(report)
            results.append(
                EnvironmentResult(
                    name=environment.name,
                    python=environment.python,
                    ok=True,
                    elapsed=elapsed,
                    report_path=report_path,
                )
            )
            renderer.environment_done(environment, elapsed, report_path)

    aggregate_path = None
    if environment_reports:
        aggregate_path = report_dir / f"validation-{run_id}.json"
        aggregate = aggregate_report(
            environment_reports,
            profile_names=profile_names,
            report_path=aggregate_path,
        )
        aggregate_path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
        )
        renderer.measurement_summary(aggregate["measurements"])

    regression_code = 0
    if baseline is not None and aggregate_path is not None:
        regressions = compare_reports(baseline, aggregate_path)
        renderer.regression_summary(regressions)
        regression_code = regression_exit_code(
            regressions, fail_on_regression=fail_on_regression
        )

    renderer.run_summary(results, aggregate_path)
    failure_code = 0 if all(result.ok for result in results) else 1
    return failure_code or regression_code


def run_environment(
    environment: EnvironmentSpec,
    scenarios: list[ScenarioSpec],
    wheel_dir: Path,
    fixture_dir: Path,
    trace_dir: Path,
    work_dir: Path,
    *,
    report_dir: Path,
    run_id: str,
    keep: bool,
    verbose: bool,
    renderer: Renderer,
) -> dict[str, Any]:
    installed = prepare_installed_wheel_environment(
        environment,
        root=ROOT,
        wheel_dir=wheel_dir,
        fixture_dir=fixture_dir,
        work_dir=work_dir,
        keep=keep,
        verbose=verbose,
        renderer=renderer,
    )
    installed.env.setdefault("RUST_LOG", "warn")
    installed.env.setdefault("PYTHONUNBUFFERED", "1")

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment.name,
        "python": environment.python,
        "tier": environment.tier,
        "rlmesh": installed_rlmesh_version(
            installed.python,
            installed.logs / f"{environment.name}-version.log",
            installed.env,
            verbose=verbose,
            renderer=renderer,
        ),
        "dependencies": list(environment.dependencies),
        "scenarios": [scenario.name for scenario in scenarios],
        "traces": [],
        "external": [],
        "measurements": [],
    }

    for scenario in scenarios:
        if scenario.kind == "trace":
            trace_result = run_trace_scenario(
                environment,
                scenario,
                installed.python,
                installed.env,
                installed.logs,
                trace_dir,
                report_dir=report_dir,
                run_id=run_id,
                verbose=verbose,
                renderer=renderer,
            )
            report["traces"].append(trace_result)

    artifact_scenarios = [
        scenario for scenario in scenarios if scenario.kind == "artifact"
    ]
    if artifact_scenarios:
        artifact_report = run_artifact_scenarios(
            environment,
            artifact_scenarios,
            installed.python,
            installed.env,
            installed.logs,
            report_dir=report_dir,
            run_id=run_id,
            verbose=verbose,
            renderer=renderer,
        )
        report["measurements"] = artifact_report["measurements"]

    for scenario in scenarios:
        if scenario.kind == "external":
            run_external_scenario(
                environment,
                scenario,
                installed.env,
                installed.logs,
                verbose=verbose,
                renderer=renderer,
            )
            report["external"].append(scenario.name)

    report_path = report_dir / f"{environment.name}-{run_id}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_trace_scenario(
    environment: EnvironmentSpec,
    scenario: ScenarioSpec,
    python: Path,
    env: dict[str, str],
    logs: Path,
    trace_dir: Path,
    *,
    report_dir: Path,
    run_id: str,
    verbose: bool,
    renderer: Renderer,
) -> dict[str, Any]:
    env_log = logs / f"{environment.name}-{scenario.name}-env.log"
    server, address = start_process_until(
        env_server_command(python, scenario),
        cwd=ROOT,
        env=env,
        log_path=env_log,
        label=f"start env endpoint {scenario.name}",
        verbose=verbose,
        renderer=renderer,
        timeout_seconds=min_timeout(scenario.timeout_seconds, environment, 60),
        predicate=parse_server_address,
    )

    trace_path = report_dir / f"{environment.name}-{scenario.name}-{run_id}.trace.json"
    driver_command = [
        str(python),
        "-m",
        "rlmesh_system_fixtures.driver",
        "trace",
        "--scenario",
        scenario.name,
        "--address",
        address,
        "--client",
        scenario.client,
        "--model",
        str(scenario.model),
        "--steps",
        str(scenario.steps),
        "--output",
        str(trace_path),
    ]
    if scenario.seed is not None:
        driver_command.extend(["--seed", str(scenario.seed)])

    try:
        run_command(
            driver_command,
            cwd=ROOT,
            env=env,
            log_path=logs / f"{environment.name}-{scenario.name}-driver.log",
            label=f"run trace scenario {scenario.name}",
            verbose=verbose,
            renderer=renderer,
            timeout_seconds=scenario.timeout_seconds or environment.timeout_seconds,
        )
        trace_result = {
            "name": scenario.name,
            "trace_path": str(trace_path),
            "address": address,
        }
        if scenario.trace is not None:
            baseline_path = trace_dir / scenario.trace
            assert_trace_matches(
                baseline_path,
                trace_path,
                logs / f"{environment.name}-{scenario.name}-trace.log",
            )
            trace_result["baseline_path"] = str(baseline_path)
        return trace_result
    finally:
        server.stop()


def run_artifact_scenarios(
    environment: EnvironmentSpec,
    scenarios: list[ScenarioSpec],
    python: Path,
    env: dict[str, str],
    logs: Path,
    *,
    report_dir: Path,
    run_id: str,
    verbose: bool,
    renderer: Renderer,
) -> dict[str, Any]:
    report_path = report_dir / f"{environment.name}-artifacts-{run_id}.json"
    command = [
        str(python),
        "-m",
        "rlmesh_system_fixtures.artifacts",
        "--environment",
        environment.name,
        "--python-version",
        environment.python,
        "--tier",
        environment.tier,
        "--samples",
        str(environment.samples),
        "--warmups",
        str(environment.warmups),
        "--output",
        str(report_path),
    ]
    for dependency in environment.dependencies:
        command.extend(["--dependency", dependency])
    for scenario in scenarios:
        assert scenario.artifact is not None
        command.extend(["--artifact", scenario.artifact])

    run_command(
        command,
        cwd=ROOT,
        env=env,
        log_path=logs / f"{environment.name}-artifacts.log",
        label="run artifact scenarios",
        verbose=verbose,
        renderer=renderer,
        timeout_seconds=environment.timeout_seconds,
    )
    return json.loads(report_path.read_text())


def run_external_scenario(
    environment: EnvironmentSpec,
    scenario: ScenarioSpec,
    env: dict[str, str],
    logs: Path,
    *,
    verbose: bool,
    renderer: Renderer,
) -> None:
    assert scenario.command_env is not None
    command_text = env.get(scenario.command_env) or os.environ.get(scenario.command_env)
    if not command_text:
        raise CommandError(
            [scenario.command_env],
            2,
            logs / f"{environment.name}-{scenario.name}.log",
            (
                f"external scenario {scenario.name!r} requires "
                f"{scenario.command_env} to contain the command to run\n"
            ),
        )

    run_command(
        shlex.split(command_text),
        cwd=ROOT,
        env=env,
        log_path=logs / f"{environment.name}-{scenario.name}.log",
        label=f"run external scenario {scenario.name}",
        verbose=verbose,
        renderer=renderer,
        timeout_seconds=scenario.timeout_seconds or environment.timeout_seconds,
    )


def env_server_command(python: Path, scenario: ScenarioSpec) -> list[str]:
    env_spec = scenario.env
    mode = str(env_spec.get("mode", "entrypoint"))
    kwargs_payload: object | None = env_spec.get("kwargs")
    command = [
        str(python),
        "-u",
        "-m",
        "rlmesh._cli.serve_env",
        "--transport",
        "tcp",
    ]
    if env_spec.get("address") is not None:
        command.extend(["--address", str(env_spec["address"])])

    if env_spec.get("fixture") is not None:
        command.extend(["--entrypoint", "rlmesh_system_fixtures.registry:make_env"])
        fixture_kwargs: dict[str, object] = {"fixture": str(env_spec["fixture"])}
        if env_spec.get("kwargs") is not None:
            fixture_kwargs["kwargs"] = env_spec["kwargs"]
        kwargs_payload = fixture_kwargs
    elif env_spec.get("gym") is not None:
        command.extend(["--env", str(env_spec["gym"])])
    elif mode == "entrypoint":
        command.extend(["--entrypoint", str(env_spec["entrypoint"])])
    elif mode == "gym":
        command.extend(["--env", str(env_spec["id"])])
    else:
        raise ValueError(f"unknown env mode {mode!r} for scenario {scenario.name!r}")

    for package in env_spec.get("packages", ()):
        command.extend(["--package", str(package)])
    if kwargs_payload is not None:
        command.extend(["--kwargs-json", json.dumps(kwargs_payload, sort_keys=True)])
    return command


def parse_server_address(line: str) -> str | None:
    match = SERVER_ADDRESS_RE.search(line)
    if match is None:
        return None
    return match.group(1)


def assert_trace_matches(baseline_path: Path, trace_path: Path, log_path: Path) -> None:
    if not baseline_path.exists():
        raise CommandError(
            ["compare-trace", str(baseline_path), str(trace_path)],
            2,
            log_path,
            f"missing trace baseline {baseline_path}\ncurrent trace: {trace_path}\n",
        )
    baseline = json.loads(baseline_path.read_text())
    current = json.loads(trace_path.read_text())
    if current != baseline:
        raise CommandError(
            ["compare-trace", str(baseline_path), str(trace_path)],
            1,
            log_path,
            (
                f"trace mismatch for {baseline_path.name}\n"
                f"baseline: {baseline_path}\ncurrent: {trace_path}\n"
            ),
        )
    log_path.write_text(f"trace matched {baseline_path}\n")


def min_timeout(
    scenario_timeout: float | None, environment: EnvironmentSpec, default: float
) -> float:
    candidates = [
        value
        for value in (scenario_timeout, environment.timeout_seconds, default)
        if value is not None
    ]
    return float(min(candidates))


def installed_rlmesh_version(
    python: Path,
    log_path: Path,
    env: dict[str, str],
    *,
    verbose: bool,
    renderer: Renderer,
) -> str:
    run_command(
        [
            str(python),
            "-c",
            "from importlib.metadata import version; print(version('rlmesh'))",
        ],
        env=env,
        log_path=log_path,
        label="read installed rlmesh version",
        verbose=verbose,
        renderer=renderer,
    )
    return log_path.read_text().strip() or version("rlmesh")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        raise SystemExit(130) from None
