from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

LOCK = Path(__file__).resolve().parents[3] / "scripts/dev/machine_lock.py"


def locked(
    name: str, code: str, env: dict[str, str], **kwargs: Any
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, str(LOCK), name, sys.executable, "-c", code],
        env=env,
        text=True,
        **kwargs,
    )


def test_second_job_waits_for_the_first(tmp_path: Path) -> None:
    env = dict(os.environ, RLMESH_LOCK_DIR=str(tmp_path))
    hold = "import sys; print('held', flush=True); sys.stdin.readline()"
    first = locked("heavy", hold, env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert first.stdout is not None
    assert first.stdout.readline().strip() == "held"

    second = locked(
        "heavy", "print('second')", env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert second.stderr is not None
    waiting = second.stderr.readline()
    assert "waiting for the heavy lock, held by:" in waiting
    assert str(Path.cwd()) in waiting

    first.communicate("\n")
    assert second.communicate()[0].strip() == "second"
    assert (first.returncode, second.returncode) == (0, 0)


def test_command_exit_status_passes_through(tmp_path: Path) -> None:
    env = dict(os.environ, RLMESH_LOCK_DIR=str(tmp_path))
    job = locked("heavy", "raise SystemExit(7)", env)
    assert job.wait() == 7


def test_rejects_bad_lock_names(tmp_path: Path) -> None:
    env = dict(os.environ, RLMESH_LOCK_DIR=str(tmp_path))
    job = locked("../escape", "pass", env, stderr=subprocess.PIPE)
    assert job.communicate()[1].startswith("usage:")
    assert job.returncode == 1
    assert not list(tmp_path.iterdir())


def test_nested_command_reuses_the_inherited_lock(tmp_path: Path) -> None:
    env = dict(os.environ, RLMESH_LOCK_DIR=str(tmp_path))
    command = [
        sys.executable,
        str(LOCK),
        "heavy",
        sys.executable,
        "-c",
        "print('nested')",
    ]
    job = locked(
        "heavy",
        f"import os; os.execv({sys.executable!r}, {command!r})",
        env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        output, error = job.communicate(timeout=10)
    finally:
        if job.poll() is None:
            job.kill()
            job.communicate()
    assert job.returncode == 0, error
    assert output.strip() == "nested"
    assert "waiting" not in error


def test_stale_inherited_descriptor_still_acquires_the_lock(tmp_path: Path) -> None:
    env = dict(
        os.environ, RLMESH_LOCK_DIR=str(tmp_path), RLMESH_MACHINE_LOCK_HEAVY_FD="9999"
    )
    job = locked("heavy", "print('acquired')", env, stdout=subprocess.PIPE)
    assert job.communicate(timeout=10)[0].strip() == "acquired"
    assert job.returncode == 0
    assert (tmp_path / "heavy.holder").exists()


def test_missing_nested_command_fails_without_waiting(tmp_path: Path) -> None:
    env = dict(os.environ, RLMESH_LOCK_DIR=str(tmp_path))
    command = [sys.executable, str(LOCK), "heavy", str(tmp_path / "missing")]
    job = locked(
        "heavy",
        f"import os; os.execv({sys.executable!r}, {command!r})",
        env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _, error = job.communicate(timeout=10)
    finally:
        if job.poll() is None:
            job.kill()
            job.communicate()
    assert job.returncode == 1
    assert "cannot run" in error
    assert "waiting" not in error
