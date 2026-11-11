"""A served env or model outlives a driver that vanishes mid-run.

The driver is killed (SIGKILL) in the middle of an episode or a predict, or
its link is cut under it; the served process keeps serving, and the next
driver runs to completion, an env from a clean reset.
"""

from __future__ import annotations

import os
import signal
import socket
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
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

EPISODE = 8

SERVED = textwrap.dedent(
    """
    import os
    import time

    import gymnasium as gym
    import numpy as np
    import rlmesh.numpy
    from rlmesh import EnvFactory


    def mark(text):
        with open(os.environ["MARKS"], "a") as marks:
            marks.write(text + "\\n")


    class CountingEnv(gym.Env):
        observation_space = gym.spaces.Box(0.0, 100.0, (1,), np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 0
            mark(f"reset {seed}")
            return np.zeros(1, np.float32), {}

        def step(self, action):
            self.t += 1
            mark(f"step {self.t}")
            time.sleep(0.15)
            done = self.t >= int(os.environ["EPISODE"])
            return np.full(1, self.t, np.float32), 1.0, False, done, {}


    class Env(EnvFactory):
        def make(self, **kwargs):
            return CountingEnv()


    class Policy(rlmesh.numpy.Model):
        def predict(self, observation):
            mark("predict")
            time.sleep(0.15)
            return 0
    """
)

DRIVER = textwrap.dedent(
    """
    import sys

    import numpy as np
    import rlmesh
    from served import CountingEnv

    kind, address, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        if kind == "env":
            env = rlmesh.RemoteEnv(address, connect_timeout_seconds=30)
            observation, _ = env.reset(seed=seed)
            steps = 0
            while True:
                observation, _, terminated, truncated, _ = env.step(0)
                steps += 1
                if terminated or truncated:
                    break
            print(f"DONE steps={steps} last={float(np.asarray(observation)[0])}", flush=True)
        else:
            result = rlmesh.numpy.Model(address).run(CountingEnv(), seeds=[seed])
            print(f"DONE reward={result.mean_reward}", flush=True)
    except Exception as exc:
        print(f"DRIVER {type(exc).__name__}: {exc}", flush=True)
    """
)


def _free_base() -> int:
    from rlmesh._rlmesh import sandbox_session_ports

    return sandbox_session_ports()


@final
class _Relay:
    """Four relayed session ports, whose connections can be cut."""

    def __init__(self, target: int) -> None:
        self.base = _free_base()
        self.sockets: list[socket.socket] = []
        self.lock = threading.Lock()
        for offset in range(4):
            listener = socket.create_server(("127.0.0.1", self.base + offset))
            self.sockets.append(listener)
            threading.Thread(
                target=self._accept, args=(listener, target + offset), daemon=True
            ).start()

    def _accept(self, listener: socket.socket, port: int) -> None:
        while True:
            try:
                client = listener.accept()[0]
                upstream = socket.create_connection(("127.0.0.1", port))
            except OSError:
                return
            with self.lock:
                self.sockets += [client, upstream]
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    @staticmethod
    def _pump(source: socket.socket, sink: socket.socket) -> None:
        try:
            while data := source.recv(65536):
                sink.sendall(data)
        except OSError:
            pass

    def cut(self) -> None:
        with self.lock:
            for sock in self.sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()


@final
class _Served:
    def __init__(self, tmp: Path, kind: str) -> None:
        from rlmesh import launch

        _ = (tmp / "served.py").write_text(SERVED, encoding="utf-8")
        _ = (tmp / "driver.py").write_text(DRIVER, encoding="utf-8")
        self.tmp, self.kind = tmp, kind
        self.marks = tmp / "marks"
        self.marks.touch()
        identity, _ = launch.launch_identity()
        self.port = _free_base()
        name = "env" if kind == "env" else "model"
        self.address = launch.launched_address(identity, f"127.0.0.1:{self.port}", name)
        self.env = {
            **os.environ,
            "PYTHONPATH": str(tmp),
            "MARKS": str(self.marks),
            "EPISODE": str(EPISODE),
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
                f"127.0.0.1:{self.port}",
                *target,
            ],
            cwd=tmp,
            env={**self.env, "RLMESH_HOST_IDENTITY": identity},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.log: list[str] = []
        threading.Thread(target=self._read, daemon=True).start()
        self.wait(lambda: any("RLMesh serving" in line for line in self.log))

    def _read(self) -> None:
        assert self.server.stdout is not None
        for line in self.server.stdout:
            self.log.append(line)

    def wait(self, ready: Callable[[], bool], timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while not ready():
            if self.server.poll() is not None or time.monotonic() > deadline:
                pytest.fail("".join(self.log))
            time.sleep(0.05)

    def marks_text(self) -> str:
        return self.marks.read_text(encoding="utf-8")

    def driver(self, seed: int, address: str | None = None) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [
                sys.executable,
                str(self.tmp / "driver.py"),
                self.kind,
                address or self.address,
                str(seed),
            ],
            cwd=self.tmp,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def mid_run(self) -> None:
        call = "step" if self.kind == "env" else "predict"
        self.wait(lambda: self.marks_text().count(call) >= 2)

    def close(self) -> None:
        if self.server.poll() is None:
            self.server.send_signal(signal.SIGTERM)
            _ = self.server.wait(timeout=20)


def _run_to_completion(served: _Served, seed: int) -> str:
    out, _ = served.driver(seed).communicate(timeout=90)
    assert out.startswith("DONE"), out + "".join(served.log)
    assert served.server.poll() is None, "".join(served.log)
    return out


@pytest.mark.parametrize("kind", ["env", "model"])
def test_a_killed_driver_leaves_the_server_serving(tmp_path: Path, kind: str) -> None:
    served = _Served(tmp_path, kind)
    try:
        first = served.driver(seed=1)
        served.mid_run()
        first.kill()
        _ = first.wait()
        out = _run_to_completion(served, seed=2)
        if kind == "env":
            # A clean reset: the abandoned episode's count does not carry.
            assert out == f"DONE steps={EPISODE} last={float(EPISODE)}\n", out
            assert "reset 2\n" in served.marks_text()
    finally:
        served.close()


def test_an_env_whose_link_is_cut_mid_episode_serves_the_next_driver(
    tmp_path: Path,
) -> None:
    served = _Served(tmp_path, "env")
    relay = _Relay(served.port)
    try:
        relayed = served.address.replace(f":{served.port}#", f":{relay.base}#")
        first = served.driver(seed=1, address=relayed)
        served.mid_run()
        relay.cut()
        out, _ = first.communicate(timeout=60)
        assert "went away" in out, out
        out = _run_to_completion(served, seed=2)
        assert out == f"DONE steps={EPISODE} last={float(EPISODE)}\n", out
    finally:
        served.close()
