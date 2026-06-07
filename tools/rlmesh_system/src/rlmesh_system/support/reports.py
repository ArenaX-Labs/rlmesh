from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentResult:
    name: str
    python: str
    ok: bool
    elapsed: float
    report_path: Path | None = None
    log_path: Path | None = None
    returncode: int | None = None


@dataclass(frozen=True)
class RegressionResult:
    environment: str
    measurement: str
    status: str
    current_ms: float | None
    baseline_ms: float | None
    reason: str


def aggregate_report(
    reports: list[dict[str, Any]], *, profile_names: list[str], report_path: Path
) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for report in reports:
        coverage.append(
            {
                "environment": report["environment"],
                "python": report["python"],
                "tier": report["tier"],
                "scenarios": report["scenarios"],
                "dependencies": report["dependencies"],
            }
        )
        for measurement in report.get("measurements", []):
            measurements.append(
                {
                    "environment": report["environment"],
                    "python": report["python"],
                    "rlmesh": report["rlmesh"],
                    **measurement,
                }
            )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profiles": profile_names,
        "report_path": str(report_path),
        "coverage": coverage,
        "environments": reports,
        "measurements": measurements,
    }


def compare_reports(baseline_path: Path, current_path: Path) -> list[RegressionResult]:
    baseline = measurement_index(load_measurements(baseline_path))
    current = measurement_index(load_measurements(current_path))
    results: list[RegressionResult] = []

    for key, current_measurement in current.items():
        baseline_measurement = baseline.get(key)
        environment, name = key
        if baseline_measurement is None:
            results.append(
                RegressionResult(
                    environment=environment,
                    measurement=name,
                    status="new",
                    current_ms=float(current_measurement["median_ms"]),
                    baseline_ms=None,
                    reason="no baseline measurement",
                )
            )
            continue
        results.append(compare_measurement(current_measurement, baseline_measurement))

    for key, baseline_measurement in baseline.items():
        if key in current:
            continue
        environment, name = key
        results.append(
            RegressionResult(
                environment=environment,
                measurement=name,
                status="missing",
                current_ms=None,
                baseline_ms=float(baseline_measurement["median_ms"]),
                reason="measurement is absent from current report",
            )
        )

    return results


def compare_measurement(
    current: dict[str, Any], baseline: dict[str, Any]
) -> RegressionResult:
    environment = str(current["environment"])
    name = str(current["name"])
    current_ms = float(current["median_ms"])
    baseline_ms = float(baseline["median_ms"])
    relative, absolute_ms, throughput_drop = thresholds_for(name)
    delta = current_ms - baseline_ms
    relative_limit = baseline_ms * (1.0 + relative)

    if current_ms > relative_limit and delta > absolute_ms:
        return RegressionResult(
            environment=environment,
            measurement=name,
            status="regression",
            current_ms=current_ms,
            baseline_ms=baseline_ms,
            reason=(
                f"median increased by {delta:.4f}ms "
                f"({current_ms / baseline_ms - 1.0:.1%})"
            ),
        )

    current_throughput = current.get("throughput_mib_s")
    baseline_throughput = baseline.get("throughput_mib_s")
    if (
        throughput_drop is not None
        and isinstance(current_throughput, int | float)
        and isinstance(baseline_throughput, int | float)
        and current_throughput < baseline_throughput * (1.0 - throughput_drop)
    ):
        return RegressionResult(
            environment=environment,
            measurement=name,
            status="regression",
            current_ms=current_ms,
            baseline_ms=baseline_ms,
            reason=(
                f"throughput dropped from {baseline_throughput:.2f} "
                f"to {current_throughput:.2f} MiB/s"
            ),
        )

    return RegressionResult(
        environment=environment,
        measurement=name,
        status="ok",
        current_ms=current_ms,
        baseline_ms=baseline_ms,
        reason="within warn-first threshold",
    )


def thresholds_for(name: str) -> tuple[float, float, float | None]:
    if name.startswith(("tensor.numpy.asarray", "tensor.torch.as_tensor")):
        return 0.15, 0.05, None
    if name.startswith("remote.image"):
        return 0.20, 0.25, 0.15
    if name.startswith(("remote.mujoco", "remote.sai-mujoco")):
        return 0.25, 1.0, None
    if name.startswith("remote.small"):
        return 0.20, 0.25, None
    return 0.20, 0.25, None


def regression_exit_code(
    results: list[RegressionResult], *, fail_on_regression: bool
) -> int:
    if not fail_on_regression:
        return 0
    return 1 if any(result.status == "regression" for result in results) else 0


def measurement_index(
    measurements: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(measurement["environment"]), str(measurement["name"])): measurement
        for measurement in measurements
    }


def load_measurements(path: Path) -> list[dict[str, Any]]:
    report = json.loads(path.read_text())
    if "measurements" not in report:
        raise SystemExit(f"{path} is not an RLMesh validation report")

    measurements = report["measurements"]
    if measurements and "environment" in measurements[0]:
        return measurements

    environment = report.get("environment")
    python = report.get("python")
    rlmesh = report.get("rlmesh")
    return [
        {
            "environment": environment,
            "python": python,
            "rlmesh": rlmesh,
            **measurement,
        }
        for measurement in measurements
    ]
