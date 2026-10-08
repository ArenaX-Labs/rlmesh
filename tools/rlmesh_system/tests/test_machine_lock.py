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
