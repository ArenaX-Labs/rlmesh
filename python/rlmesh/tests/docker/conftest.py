"""The sandbox Docker gate's images, built once, and its timing report."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sandbox_gate import (
    LABEL,
    TIMINGS,
    GateImages,
    build_images,
    check_same_build,
    docker,
    docker_missing,
    remove_labelled,
    run_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(scope="session")
def gate_images() -> Iterator[GateImages]:
    """The gate's images; this run's containers and images go at the end."""
    if missing := docker_missing():
        if os.environ.get("RLMESH_DOCKER_GATE_REQUIRED") == "1":
            pytest.fail(f"sandbox Docker gate required, but {missing}")
        pytest.skip(f"sandbox Docker gate skipped: {missing}")
    run = run_id()
    try:
        images = build_images(run)
        check_same_build(images)
        yield images
    finally:
        if os.environ.get("RLMESH_DOCKER_GATE_KEEP_IMAGES") == "1":
            containers = docker("ps", "-aq", "--filter", f"label={LABEL}={run}").split()
            if containers:
                docker("rm", "-f", *containers)
        else:
            remove_labelled(run)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if not TIMINGS:
        return
    terminalreporter.section("sandbox Docker gate timings (seconds)")
    rows: dict[str, dict[str, float]] = {}
    for what, values in sorted(TIMINGS.items()):
        ordered = sorted(values)
        row = {
            "n": len(ordered),
            "min": ordered[0],
            "median": statistics.median(ordered),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": ordered[-1],
        }
        rows[what] = row
        terminalreporter.write_line(
            f"{what:<44} n={row['n']:<3} min {row['min']:.3f}  "
            f"median {row['median']:.3f}  p90 {row['p90']:.3f}  max {row['max']:.3f}"
        )
    if report := os.environ.get("RLMESH_DOCKER_GATE_REPORT"):
        Path(report).write_text(
            json.dumps({"summary": rows, "samples": TIMINGS}, indent=2)
        )
