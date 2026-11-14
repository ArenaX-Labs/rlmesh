"""The happy path says nothing at warn or above, and a served peer says when it is ready.

A complete local run, a served env driven by ``RemoteEnv``, and a driver
stopped with Ctrl-C each end without a warning or an error line on either
side: no undeclared-edition warning, no shutdown noise, no third-party
source paths. ``RLMESH_READY_FILE`` appears once the served env answers and
goes when it stops, without the session's key.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("numpy")
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

SERVED = textwrap.dedent(
    """
    import os
    import time

    import gymnasium as gym
    import numpy as np
    from rlmesh import EnvFactory


    class StepEnv(gym.Env):
        observation_space = gym.spaces.Box(0.0, 100.0, (1,), np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            return np.zeros(1, np.float32), {}

        def step(self, action):
            self.t += 1
            time.sleep(float(os.environ.get("STEP_SECONDS", "0")))
            return np.full(1, self.t, np.float32), 1.0, False, self.t >= 3, {}


    class Env(EnvFactory):
        def make(self, **kwargs):
            return StepEnv()
    """
)

LOCAL_RUN = textwrap.dedent(
    """
    import rlmesh.numpy
    from served import StepEnv

    result = rlmesh.numpy.Model(lambda observation: 0).run(StepEnv(), episodes=2)
    print(f"DONE {len(result.episodes)}", flush=True)
    """
)

DRIVER = textwrap.dedent(
    """
    import sys

    import rlmesh

    try:
        env = rlmesh.RemoteEnv(sys.argv[1], connect_timeout_seconds=30)
        for episode in range(int(sys.argv[2])):
            env.reset(seed=episode)
            print("EPISODE", flush=True)
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(0)
                done = terminated or truncated
        env.close()
        print("DONE", flush=True)
    except KeyboardInterrupt:
        print("INTERRUPTED", flush=True)
    """
)

#: A tracing line at warn or above, a Python warning, or a source path from
#: a dependency's build.
NOISE = re.compile(r"\b(WARN|ERROR)\b|Warning:|\.cargo/registry|panicked at")


def _quiet(output: str) -> None:
    noisy = [line for line in output.splitlines() if NOISE.search(line)]
    assert noisy == [], output


def _environment(tmp: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHONPATH": str(tmp),
        "RLMESH_STATE_DIR": str(tmp / "state"),
    }
    # The suite floats deliberately; the happy path must be quiet without it.
    for name in ("RLMESH_WORKFLOW_EDITION", "RUST_LOG"):
        environment.pop(name, None)
    return environment


@pytest.fixture
def project(tmp_path: Path) -> Path:
    _ = (tmp_path / "served.py").write_text(SERVED, encoding="utf-8")
    _ = (tmp_path / "local_run.py").write_text(LOCAL_RUN, encoding="utf-8")
    _ = (tmp_path / "driver.py").write_text(DRIVER, encoding="utf-8")
    return tmp_path


def test_a_local_run_is_quiet(project: Path) -> None:
    done = subprocess.run(
        [sys.executable, "local_run.py"],
        cwd=project,
        env=_environment(project),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = done.stdout + done.stderr
    assert done.returncode == 0 and "DONE 2" in output, output
    _quiet(output)


class _Served:
    """``python -m rlmesh.serve --env served:Env`` on a launched address."""

    def __init__(self, project: Path, step_seconds: float = 0.0) -> None:
        from rlmesh import launch
        from rlmesh._rlmesh import sandbox_session_port

        identity, _ = launch.launch_identity()
        port = sandbox_session_port()
        self.address = launch.launched_address(identity, f"127.0.0.1:{port}", "env")
        self.ready = project / "ready" / "env.json"
        self.ready.parent.mkdir()
        self.environment = {
            **_environment(project),
            "STEP_SECONDS": str(step_seconds),
        }
        self.log = project / "served.log"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rlmesh.serve",
                "--env",
                "served:Env",
                "--address",
                f"127.0.0.1:{port}",
            ],
            cwd=project,
            env={
                **self.environment,
                "RLMESH_HOST_IDENTITY": identity,
                "RLMESH_READY_FILE": str(self.ready),
            },
            stdout=self.log.open("w"),
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 60
        while not self.ready.exists():
            if self.process.poll() is not None or time.monotonic() > deadline:
                pytest.fail(self.output())
            time.sleep(0.05)

    def output(self) -> str:
        return self.log.read_text(encoding="utf-8")

    def driver(self, project: Path, episodes: int) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, "driver.py", self.address, str(episodes)],
            cwd=project,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            assert self.process.wait(timeout=30) == 0, self.output()


def test_a_served_env_and_its_driver_are_quiet_and_ready_says_so(
    project: Path,
) -> None:
    served = _Served(project)
    try:
        ready = json.loads(served.ready.read_text(encoding="utf-8"))
        assert set(ready) == {"kind", "participant", "address", "port", "pid"}
        assert ready["kind"] == "env"
        assert ready["participant"] == "env"
        assert ready["pid"] == served.process.pid
        key = served.address.split("#", 1)[1].split("/", 1)[0]
        assert key not in served.ready.read_text(encoding="utf-8")
        assert "#…/env" in ready["address"]
        # The serving line comes once the env answers, with the same address.
        assert f"RLMesh serving env on {ready['address']} " in served.output()

        out, _ = served.driver(project, episodes=2).communicate(timeout=120)
        assert "DONE" in out, out + served.output()
        _quiet(out)
    finally:
        served.stop()
    assert not served.ready.exists()
    _quiet(served.output())


def test_ctrl_c_of_a_driver_is_quiet_on_both_sides(project: Path) -> None:
    served = _Served(project, step_seconds=0.2)
    try:
        driver = served.driver(project, episodes=50)
        assert driver.stdout is not None
        assert driver.stdout.readline().strip() == "EPISODE"
        time.sleep(0.5)
        driver.send_signal(signal.SIGINT)
        out, _ = driver.communicate(timeout=60)
        assert "INTERRUPTED" in out, out
        _quiet(out)
        # The env keeps serving the next driver.
        out, _ = served.driver(project, episodes=1).communicate(timeout=120)
        assert "DONE" in out, out + served.output()
    finally:
        served.stop()
    _quiet(served.output())
