"""EnvServer behaviour aligned with the native servers (ENV-P1, ENV-P2, ENV-P7)."""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np
import pytest
import rlmesh


class _Env:
    observation_space = gym.spaces.Discrete(4)
    action_space = gym.spaces.Discrete(2)

    def __init__(self, close_error: Exception | None = None):
        self.close_error = close_error
        self.server: rlmesh.EnvServer | None = None

    def reset(self, *, seed=None, options=None):
        return 0, {}

    def step(self, action):
        if self.server is not None:
            # From inside its own callback: only triggers the drain.
            self.server.shutdown()
        return 1, 0.0, False, False, {}

    def close(self):
        if self.close_error is not None:
            raise self.close_error


def _connect(address: str) -> rlmesh.RemoteEnv:
    deadline = time.monotonic() + 5
    while True:
        try:
            return rlmesh.RemoteEnv(address)
        except ConnectionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.02)


def test_a_failed_close_is_an_rlmesh_exception_with_the_serve_prefix():
    server = rlmesh.EnvServer(_Env(close_error=ValueError("boom")), "127.0.0.1:0")
    server.start()
    with pytest.raises(rlmesh.RLMeshException) as raised:
        server.shutdown()
    assert str(raised.value) == "serve failed: close failed: boom"


def test_a_foreground_serve_failure_has_the_serve_prefix():
    server = rlmesh.EnvServer(_Env(close_error=ValueError("boom")), "127.0.0.1:0")
    result: list[BaseException] = []

    def serve() -> None:
        try:
            server.serve()
        except BaseException as exc:
            result.append(exc)

    # A worker-thread serve, stopped by its owner from another thread.
    worker = threading.Thread(target=serve)
    worker.start()
    time.sleep(0.3)
    server.shutdown()
    worker.join(10)
    assert len(result) == 1
    assert isinstance(result[0], rlmesh.RLMeshException)
    assert str(result[0]) == "serve failed: close failed: boom"


def test_shutdown_from_a_callback_under_start_only_triggers_the_drain():
    env = _Env()
    server = rlmesh.EnvServer(env, "127.0.0.1:0")
    env.server = server
    server.start()
    remote = _connect(server.address)
    remote.reset(seed=0)
    remote.step(0)
    assert server.wait(10) is True
    remote.close()


class _Vector:
    """A two-lane vector env recording the thread each call runs on."""

    num_envs = 2
    single_observation_space = gym.spaces.Discrete(4)
    single_action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.threads: list[int] = []

    def reset(self, *, seed=None, options=None):
        self.threads.append(threading.get_ident())
        return np.zeros(2, dtype=np.int64), {}

    def step(self, actions):
        self.threads.append(threading.get_ident())
        zeros = np.zeros(2, dtype=np.int64)
        return zeros, np.zeros(2), np.zeros(2, dtype=bool), np.zeros(2, dtype=bool), {}

    def close(self):
        self.threads.append(threading.get_ident())


def test_a_background_vector_env_runs_on_one_thread():
    """ENV-P7: one dedicated thread, never the blocking pool."""
    env = _Vector()
    server = rlmesh.EnvServer(env, "127.0.0.1:0")
    server.start()
    remote = rlmesh.RemoteVectorEnv(server.address)
    remote.reset(seed=[0, 1])
    for _ in range(5):
        remote.step([0, 1])
    remote.close()
    server.shutdown()
    assert len(env.threads) == 7
    assert len(set(env.threads)) == 1
    assert env.threads[0] != threading.get_ident()
