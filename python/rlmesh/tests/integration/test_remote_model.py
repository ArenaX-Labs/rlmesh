"""RemoteModel: drive a served policy against a served env in the symmetric loop."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, cast

import pytest


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])
    finally:
        sock.close()


class TinyEnv:
    def __init__(self) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[int, dict[str, object]]:
        _ = seed, options
        return 0, {}

    def step(self, action: object) -> tuple[int, float, bool, bool, dict[str, object]]:
        return 1, 1.0, True, False, {"action": action}

    def close(self) -> None:
        return None


def _serve_env(env: object) -> Any:
    import rlmesh
    from rlmesh._server import EnvLike as ServedEnv

    try:
        server = rlmesh.EnvServer(cast("ServedEnv", env), host="127.0.0.1", port=0)
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    return server


def _serve_model(address: str, seen: list[object]) -> threading.Thread:
    import rlmesh

    def predict(observation: object) -> int:
        seen.append(observation)
        return 1

    def run() -> None:
        rlmesh.Model(predict).serve(
            address, options=rlmesh.ServeOptions(allow_remote_shutdown=True)
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _connect_model_with_retry(address: str, env: object) -> Any:
    import rlmesh

    deadline = time.monotonic() + 5.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return rlmesh.RemoteModel(address).against(env)
        except Exception as exc:  # retry until the server is up
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"model server at {address} never came up") from last_error


def test_remote_model_drives_a_served_env_in_the_symmetric_loop() -> None:
    import rlmesh

    seen: list[object] = []
    env_server = _serve_env(TinyEnv())
    model_address = f"127.0.0.1:{_free_port()}"
    _serve_model(model_address, seen)

    try:
        env = rlmesh.RemoteEnv(env_server.address)
        model = _connect_model_with_retry(model_address, env)

        obs, _info = env.reset(seed=0)
        model.reset()
        done = False
        steps = 0
        while not done and steps < 5:
            action = model.predict(obs)
            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            steps += 1

        # The policy was consulted, returned its action (1), and the env stepped.
        assert seen, "served policy predict was never called"
        assert steps == 1
        model.close()
        env.close()
    finally:
        env_server.shutdown()


def test_against_requires_an_env_contract() -> None:
    import rlmesh

    model = rlmesh.RemoteModel(f"127.0.0.1:{_free_port()}")
    with pytest.raises(TypeError, match="env_contract"):
        model.against(object())
