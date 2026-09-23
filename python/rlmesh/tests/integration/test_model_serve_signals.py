"""A served model stops on SIGINT the way a served env does."""

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


@pytest.mark.skipif(not hasattr(signal, "SIGINT"), reason="no SIGINT")
def test_a_served_model_exits_on_sigint(tmp_path: object) -> None:
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
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = proc.stdout.readline() if proc.stdout else ""
            if "RLMesh serving model" in line:
                break
            if proc.poll() is not None:
                pytest.fail(f"server exited early: {line}")
        else:
            pytest.fail("server never reported serving")
        proc.send_signal(signal.SIGINT)
        code = proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert code == 130
