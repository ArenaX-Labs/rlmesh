"""A served model stops on SIGINT and SIGTERM the way a served env does."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

pytest.importorskip("numpy")

MODEL_SOURCE = """
import rlmesh.numpy

class Policy(rlmesh.numpy.Model):
    def predict(self, observation):
        return 0
"""


def _catches(pid: int, signum: int) -> bool:
    """Whether ``pid`` has a handler for ``signum`` (Linux; elsewhere, after a beat)."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as status:
            caught = next(line for line in status if line.startswith("SigCgt:"))
    except OSError:
        time.sleep(1)
        return True
    return bool(int(caught.split()[1], 16) >> (signum - 1) & 1)


def _serve(tmp_path: object) -> subprocess.Popen[str]:
    module = os.path.join(str(tmp_path), "sigmod.py")
    with open(module, "w", encoding="utf-8") as handle:
        handle.write(MODEL_SOURCE)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "rlmesh.serve",
            "sigmod:Policy",
            "--address",
            "127.0.0.1:0",
        ],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if "RLMesh serving model" in line:
            return proc
        if proc.poll() is not None:
            pytest.fail(f"server exited early: {line}")
    proc.kill()
    pytest.fail("server never reported serving")


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="no SIGINT")
def test_a_served_model_exits_on_sigint(tmp_path: object) -> None:
    proc = _serve(tmp_path)
    try:
        proc.send_signal(signal.SIGINT)
        code = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert code == 130


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="no SIGTERM")
def test_a_served_model_stops_gracefully_and_promptly_on_sigterm(
    tmp_path: object,
) -> None:
    # As PID 1 in a container, a SIGTERM nobody handles is ignored until
    # `docker stop` kills the process; here, unhandled, it kills it (-15).
    proc = _serve(tmp_path)
    try:
        # The serving line comes just before the serve takes over SIGTERM.
        deadline = time.monotonic() + 10
        while not _catches(proc.pid, signal.SIGTERM) and time.monotonic() < deadline:
            time.sleep(0.05)
        started = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        code = proc.wait(timeout=10)
        elapsed = time.monotonic() - started
    finally:
        if proc.poll() is None:
            proc.kill()
    assert code == 0
    assert elapsed < 5, elapsed
