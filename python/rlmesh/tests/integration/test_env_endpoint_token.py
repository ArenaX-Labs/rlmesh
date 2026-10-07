"""Python servers and clients authenticate with endpoint tokens."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from typing import Any, cast

import pytest

# Defined inside each subprocess script too: the child has no access to this
# module's classes.
_CHILD_ENV = """
import rlmesh
from rlmesh import spaces

class Env:
    observation_space = spaces.Discrete(2)
    action_space = spaces.Discrete(2)
    def reset(self, *, seed=None, options=None):
        return 0, {}
    def step(self, action):
        return 0, 0.0, True, False, {}
    def close(self):
        pass
"""


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


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return cast(int, sock.getsockname()[1])
    finally:
        sock.close()


def _serve_env(token: str | None) -> Any:
    import rlmesh
    from rlmesh._server import EnvLike as ServedEnv

    try:
        server = rlmesh.EnvServer(
            cast("ServedEnv", TinyEnv()),
            host="127.0.0.1",
            port=0,
            options=rlmesh.ServeOptions(token=token),
        )
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    return server


def _run_child(script: str, token: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_ENV + textwrap.dedent(script)],
        env={**os.environ, "RLMESH_ENV_ENDPOINT_TOKEN": token},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_env_server_requires_serve_options_token() -> None:
    import rlmesh

    server = _serve_env("env-token")
    try:
        with pytest.raises(ConnectionError):
            rlmesh.RemoteEnv(server.address)
        with pytest.raises(ConnectionError):
            rlmesh.RemoteEnv(server.address, token="wrong-token")
        for token in ["env-token", "Bearer env-token"]:
            env = rlmesh.RemoteEnv(server.address, token=token)
            assert env.reset(seed=0)[0] == 0
            env.close()
    finally:
        server.shutdown()


def test_serve_options_token_survives_edition_declaration() -> None:
    import rlmesh
    from rlmesh._editions import serve_options_declaring

    options = serve_options_declaring(
        rlmesh.ServeOptions(token="env-token"),
        option=rlmesh.current_workflow_edition(),
    )
    assert options is not None
    assert options.token == "env-token"


def test_remote_model_sends_token() -> None:
    import rlmesh

    model_address = f"127.0.0.1:{_free_port()}"
    threading.Thread(
        target=lambda: rlmesh.Model(lambda observation: 1).serve(
            model_address,
            options=rlmesh.ServeOptions(token="model-token", idle_timeout_seconds=5),
        ),
        daemon=True,
    ).start()
    env_server = _serve_env(None)
    try:
        env = rlmesh.RemoteEnv(env_server.address)
        deadline = time.monotonic() + 5.0
        while True:
            try:
                model = rlmesh.RemoteModel(model_address, token="model-token")
                session = rlmesh.session(model, env)
                break
            except ConnectionError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        observation, _ = session.reset(seed=0)
        assert session.predict(observation) == 1
        session.close()

        with pytest.raises(ConnectionError):
            rlmesh.session(rlmesh.RemoteModel(model_address), env)
        env.close()
    finally:
        env_server.shutdown()


@pytest.mark.parametrize("token", ["", " \t\n"])
def test_env_server_rejects_blank_environment_token(token: str) -> None:
    result = _run_child('rlmesh.EnvServer(Env(), "127.0.0.1:0")', token)
    assert result.returncode != 0
    assert "RLMESH_ENV_ENDPOINT_TOKEN must not be empty" in result.stderr


def test_environment_token_overrides_options_and_reaches_run_loopback() -> None:
    result = _run_child(
        """
        server = rlmesh.EnvServer(
            Env(), "127.0.0.1:0", options=rlmesh.ServeOptions(token="direct-token")
        )
        server.start()
        try:
            for token in [None, "direct-token"]:
                try:
                    rlmesh.RemoteEnv(server.address, token=token)
                except ConnectionError:
                    pass
                else:
                    raise AssertionError(f"{token!r} must be rejected")
            rlmesh.RemoteEnv(server.address, token="deployment-token").close()
        finally:
            server.shutdown()

        # run() stands up its own loopback env server, which enforces the
        # deployment token; its in-process client must present it.
        result = rlmesh.Model(lambda observation: 0).run(Env(), episodes=1)
        assert result.num_episodes == 1
        """,
        "deployment-token",
    )
    assert result.returncode == 0, result.stderr
