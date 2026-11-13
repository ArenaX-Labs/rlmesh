"""How ``python -m rlmesh.serve`` ends on a signal, with a driver in a call.

Whoever launched a served env or model ends it with SIGTERM: the process
drains the call in progress, closes, and exits 0; the driver hears that the
server went away rather than waiting out its timeout. A second SIGTERM (a
call that will not return) ends the process at once, by the signal; so does
a second Ctrl-C, after the first one's drain.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import final

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")
pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGTERM") or sys.platform == "win32", reason="POSIX signals"
)

SERVED = textwrap.dedent(
    """
    import os
    import time

    import gymnasium as gym
    import numpy as np
    import rlmesh.numpy
    from rlmesh import EnvFactory


    def slow():
        time.sleep(float(os.environ["SLOW"]))


    class SlowEnv(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(2, np.float32), {}

        def step(self, action):
            with open(os.environ["MARKS"], "a") as marks:
                marks.write("step\\n")
            slow()
            return np.zeros(2, np.float32), 1.0, False, False, {}

        def close(self):
            with open(os.environ["MARKS"], "a") as marks:
                marks.write("closed\\n")


    class Env(EnvFactory):
        def make(self, **kwargs):
            return SlowEnv()


    class Policy(rlmesh.numpy.Model):
        def predict(self, observation):
            with open(os.environ["MARKS"], "a") as marks:
                marks.write("predict\\n")
            slow()
            return 0
    """
)

DRIVER = textwrap.dedent(
    """
    import sys

    import rlmesh
    from served import SlowEnv

    kind, address = sys.argv[1], sys.argv[2]
    try:
        if kind == "env":
            env = rlmesh.RemoteEnv(address, connect_timeout_seconds=30)
            env.reset(seed=0)
            while True:
                env.step(0)
        else:
            rlmesh.numpy.Model(address).run(SlowEnv(), seeds=[0])
    except Exception as exc:
        print(f"DRIVER {type(exc).__name__}: {exc}", flush=True)
    """
)


@final
class Served:
    """A served env or model, and a driver in a call to it."""

    def __init__(self, tmp: Path, kind: str, slow: float) -> None:
        from rlmesh import launch
        from rlmesh._rlmesh import sandbox_session_port

        _ = (tmp / "served.py").write_text(SERVED, encoding="utf-8")
        _ = (tmp / "driver.py").write_text(DRIVER, encoding="utf-8")
        self.marks = tmp / "marks"
        self.marks.touch()
        identity, _ = launch.launch_identity()
        port = sandbox_session_port()
        name = "env" if kind == "env" else "model"
        address = launch.launched_address(identity, f"127.0.0.1:{port}", name)
        env = {
            **os.environ,
            "PYTHONPATH": str(tmp),
            "MARKS": str(self.marks),
            "SLOW": str(slow),
            "RLMESH_WORKFLOW_EDITION": "",
            "RLMESH_STATE_DIR": str(tmp / "state"),
        }
        target = ["--env", "served:Env"] if kind == "env" else ["served:Policy"]
        self.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rlmesh.serve",
                "--address",
                f"127.0.0.1:{port}",
                *target,
            ],
            cwd=tmp,
            env={**env, "RLMESH_HOST_IDENTITY": identity},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.server_out: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        self._wait(lambda: any("RLMesh serving" in line for line in self.server_out))
        self.driver = subprocess.Popen(
            [sys.executable, str(tmp / "driver.py"), kind, address],
            cwd=tmp,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        call = "step" if kind == "env" else "predict"
        self._wait(lambda: call in self.marks.read_text(encoding="utf-8"))

    def _read(self) -> None:
        assert self.server.stdout is not None
        for line in self.server.stdout:
            self.server_out.append(line)

    def _wait(self, ready: Callable[[], bool], timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while not ready():
            if self.server.poll() is not None or time.monotonic() > deadline:
                self.close()
                pytest.fail("".join(self.server_out))
            time.sleep(0.05)

    def stop(self, signum: int, times: int = 1) -> tuple[int, float]:
        """Signal the server ``times`` times; its exit status and how long it took."""
        started = time.monotonic()
        for _ in range(times):
            self.server.send_signal(signum)
            time.sleep(0.3)
        code = self.server.wait(timeout=15)
        return code, time.monotonic() - started

    def driver_error(self) -> str:
        out, _ = self.driver.communicate(timeout=30)
        return out

    def close(self) -> None:
        for proc in (self.server, self.driver):
            if proc.poll() is None:
                proc.kill()
                _ = proc.wait()


@pytest.mark.parametrize("kind", ["env", "model"])
def test_sigterm_drains_the_call_in_progress_then_exits_zero(
    tmp_path: Path, kind: str
) -> None:
    served = Served(tmp_path, kind, slow=1.5)
    try:
        code, elapsed = served.stop(signal.SIGTERM)
        assert (code, elapsed < 5) == (0, True), (elapsed, served.server_out)
        if kind == "env":
            assert served.marks.read_text(encoding="utf-8").endswith("closed\n")
        error = served.driver_error()
        assert "went away" in error and "timed out" not in error, error
    finally:
        served.close()


@pytest.mark.parametrize("kind", ["env", "model"])
def test_a_second_sigterm_ends_a_call_that_will_not_return(
    tmp_path: Path, kind: str
) -> None:
    served = Served(tmp_path, kind, slow=60)
    try:
        code, elapsed = served.stop(signal.SIGTERM, times=2)
        assert (code, elapsed < 5) == (-signal.SIGTERM, True), elapsed
        assert "went away" in served.driver_error()
    finally:
        served.close()


def test_a_second_ctrl_c_ends_a_model_call_that_will_not_return(tmp_path: Path) -> None:
    served = Served(tmp_path, "model", slow=60)
    try:
        code, elapsed = served.stop(signal.SIGINT, times=2)
        assert (code, elapsed < 5) == (-signal.SIGINT, True), elapsed
    finally:
        served.close()
