"""Helpers for the sandbox Docker gate: its images, run label and timings.

Every container and image this suite creates carries the label
``rlmesh.test.run=<run>``, and only those are removed at the end: a shared
Docker host may run anything else. ``mise run test:sandbox:docker`` sets the
run id; see ``docs/testing.md``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
IMAGES = Path(__file__).resolve().parent / "images"
LABEL = "rlmesh.test.run"

#: Seconds, by what was timed, across the session.
TIMINGS: dict[str, list[float]] = defaultdict(list)


def run_id() -> str:
    raw = os.environ.get("RLMESH_DOCKER_GATE_RUN") or (
        f"rlmesh-gate-{ROOT.name}-{os.getpid()}"
    )
    return re.sub(r"[^a-z0-9_.-]", "-", raw.lower())


def docker_missing() -> str | None:
    """Why this machine cannot run the gate, or ``None`` when it can."""
    if shutil.which("docker") is None:
        return "the docker CLI is not on PATH"
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "the Docker daemon did not answer `docker info` within 30s"
    if proc.returncode != 0:
        return f"the Docker daemon is unreachable: {proc.stderr.strip()}"
    return None


def docker(*args: str, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=600,
        env=None if env is None else {**os.environ, **env},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def remove_labelled(run: str) -> None:
    """Remove this run's containers and images, and nothing else."""
    selector = f"label={LABEL}={run}"
    containers = docker("ps", "-aq", "--filter", selector).split()
    if containers:
        docker("rm", "-f", *containers)
    images = sorted(set(docker("images", "-q", "--filter", selector).split()))
    if images:
        docker("rmi", "-f", *images)


@dataclass(frozen=True)
class GateImages:
    run: str
    env: str
    model: str
    exits: str


def _wheel() -> Path:
    if explicit := os.environ.get("RLMESH_DOCKER_GATE_WHEEL"):
        return Path(explicit)
    arch = {"amd64": "x86_64", "arm64": "aarch64"}.get(
        platform.machine().lower(), platform.machine().lower()
    )
    found = sorted(
        (ROOT / "python/rlmesh/dist").glob(f"rlmesh-*-cp311-abi3-manylinux*{arch}*.whl")
    )
    if not found:
        pytest.fail(
            f"no cp311-abi3 manylinux {arch} wheel in python/rlmesh/dist: run "
            "`mise run build:python:docker` (the gate task does)"
        )
    return found[-1]


def _image_exists(tag: str) -> bool:
    proc = subprocess.run(
        ["docker", "image", "inspect", tag], capture_output=True, timeout=60
    )
    return proc.returncode == 0


def build_images(run: str) -> GateImages:
    """Build the env, model and failing-env images from this checkout's wheel."""
    images = GateImages(
        run=run,
        env=f"{run}-env:gate",
        model=f"{run}-model:gate",
        exits=f"{run}-exits:gate",
    )
    if os.environ.get("RLMESH_DOCKER_GATE_KEEP_IMAGES") == "1" and all(
        _image_exists(tag) for tag in (images.env, images.model, images.exits)
    ):
        return images
    wheel = _wheel()
    with tempfile.TemporaryDirectory(prefix="rlmesh-docker-gate-") as context:
        ctx = Path(context)
        shutil.copy2(wheel, ctx / wheel.name)
        examples = ROOT / "examples/python/byo_container"
        shutil.copy2(examples / "env/entrypoint.py", ctx / "env_entrypoint.py")
        shutil.copy2(examples / "model/entrypoint.py", ctx / "model_entrypoint.py")
        shutil.copy2(IMAGES / "exits.py", ctx / "exits.py")
        base = os.environ.get("RLMESH_DOCKER_GATE_BASE", "python:3.12-slim")
        for target, tag in (
            ("env", images.env),
            ("model", images.model),
            ("exits", images.exits),
        ):
            docker(
                "build",
                "--quiet",
                "--target",
                target,
                "--build-arg",
                f"BASE={base}",
                "--label",
                f"{LABEL}={run}",
                "-t",
                tag,
                "-f",
                str(IMAGES / "Dockerfile"),
                str(ctx),
            )
    return images


def check_same_build(images: GateImages) -> None:
    """Fail unless the images run this host's build: a dev build talks only
    to the identical build (its workflow edition carries the tree state)."""
    from rlmesh._rlmesh import build_info

    probe = (
        "from rlmesh._rlmesh import build_info; print(build_info().workflow_edition)"
    )
    inside = docker(
        "run",
        "--rm",
        "--label",
        f"{LABEL}={images.run}",
        "--entrypoint",
        "python",
        images.env,
        "-c",
        probe,
    )
    here = build_info().workflow_edition
    if inside != here:
        pytest.fail(
            f"the gate's images run rlmesh build {inside!r} and this host runs "
            f"{here!r}; rebuild the wheel from this tree "
            "(`mise run test:sandbox:docker --rebuild`)"
        )
