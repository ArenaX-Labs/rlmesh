from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from rlmesh_system.support.command import HarnessRenderer, run_command

RemoteT = TypeVar("RemoteT")


@dataclass(frozen=True)
class InstalledEnvironmentSpec:
    name: str
    python: str
    dependencies: tuple[str, ...]
    dependency_args: tuple[str, ...]
    env: dict[str, str]


@dataclass(frozen=True)
class InstalledEnvironment:
    python: Path
    env: dict[str, str]
    logs: Path


def prepare_installed_wheel_environment(
    spec: InstalledEnvironmentSpec,
    *,
    root: Path,
    wheel_dir: Path,
    fixture_dir: Path,
    work_dir: Path,
    extra_dependencies: tuple[str, ...] = (),
    keep: bool,
    verbose: bool,
    renderer: HarnessRenderer,
) -> InstalledEnvironment:
    venv = work_dir / "venvs" / spec.name
    logs = work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command_env = os.environ.copy()
    command_env.setdefault("UV_CACHE_DIR", str(work_dir / "uv-cache"))

    if venv.exists() and not keep:
        shutil.rmtree(venv)

    base_python = resolve_mise_python(
        spec.python,
        logs / f"{spec.name}-python.log",
        root=root,
        verbose=verbose,
        renderer=renderer,
    )
    run_command(
        [
            "uv",
            "venv",
            "--python",
            str(base_python),
            str(venv),
        ],
        cwd=root,
        env=command_env,
        log_path=logs / f"{spec.name}-venv.log",
        label="create venv",
        verbose=verbose,
        renderer=renderer,
    )

    python = venv_python(venv)
    dependencies = (*spec.dependencies, *extra_dependencies)
    if dependencies:
        run_command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                *spec.dependency_args,
                *dependencies,
            ],
            cwd=root,
            env=command_env,
            log_path=logs / f"{spec.name}-deps.log",
            label="install profile dependencies",
            verbose=verbose,
            renderer=renderer,
        )

    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--no-index",
            "--find-links",
            str(wheel_dir),
            "--reinstall-package",
            "rlmesh",
            "rlmesh",
        ],
        cwd=root,
        env=command_env,
        log_path=logs / f"{spec.name}-rlmesh.log",
        label="install rlmesh wheel",
        verbose=verbose,
        renderer=renderer,
    )

    run_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "--reinstall",
            str(fixture_dir),
        ],
        cwd=root,
        env=command_env,
        log_path=logs / f"{spec.name}-fixtures.log",
        label="install system fixtures",
        verbose=verbose,
        renderer=renderer,
    )

    assert_wheel_tag(
        python,
        spec.python,
        logs / f"{spec.name}-wheel.log",
        verbose=verbose,
        renderer=renderer,
    )

    run_env = command_env.copy()
    run_env.update(spec.env)
    return InstalledEnvironment(python=python, env=run_env, logs=logs)


def resolve_mise_python(
    python_version: str,
    log_path: Path,
    *,
    root: Path,
    verbose: bool,
    renderer: HarnessRenderer,
) -> Path:
    command = ["mise", "where", f"python@{python_version}"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.step("resolve Python interpreter")
    if verbose:
        renderer.command(command)
    process = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.write_text(process.stdout)
    if verbose and process.stdout:
        renderer.command_output(process.stdout)
    if process.returncode != 0:
        from rlmesh_system.support.command import CommandError

        raise CommandError(command, process.returncode, log_path, process.stdout)

    install_dir = Path(process.stdout.strip())
    candidates = [
        install_dir / "bin" / "python",
        install_dir / "bin" / f"python{python_version}",
        install_dir / "python.exe",
        install_dir / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise SystemExit(
        f"mise python@{python_version} is installed at {install_dir}, "
        "but no Python executable was found"
    )


def assert_wheel_tag(
    python: Path,
    python_version: str,
    log_path: Path,
    *,
    verbose: bool,
    renderer: HarnessRenderer,
) -> None:
    expected = "cp310-cp310" if python_version == "3.10" else "cp311-abi3"
    code = f"""
from importlib.metadata import distribution
wheel = distribution("rlmesh").read_text("WHEEL") or ""
tags = [line.removeprefix("Tag: ").strip() for line in wheel.splitlines() if line.startswith("Tag: ")]
print("\\n".join(tags))
if not any(tag.startswith("{expected}-") for tag in tags):
    raise SystemExit("expected wheel tag prefix {expected}, got " + ", ".join(tags))
"""
    run_command(
        [str(python), "-c", code],
        log_path=log_path,
        label="check installed wheel tag",
        verbose=verbose,
        renderer=renderer,
    )


def venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def connect_with_retry(factory: Callable[[str], RemoteT], address: str) -> RemoteT:
    deadline = time.monotonic() + 5.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return factory(address)
        except BaseException as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"failed to connect to {address}") from last_error


def env_server(env: object, options: object | None = None):
    import rlmesh

    return rlmesh.EnvServer(env, host="127.0.0.1", port=0, options=options)
