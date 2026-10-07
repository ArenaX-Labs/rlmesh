"""The Python env serving path enforces the deployment endpoint credential."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

# Each child process defines its own env: the deployment variable is read at
# bind, and setting it in this process could authenticate unrelated servers.
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


def _run_child(script: str, token: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_ENV + textwrap.dedent(script)],
        env={**os.environ, "RLMESH_ENV_ENDPOINT_TOKEN": token},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("token", ["", " \t\n"])
def test_env_server_rejects_blank_environment_token(token: str) -> None:
    result = _run_child('rlmesh.EnvServer(Env(), "127.0.0.1:0")', token)
    assert result.returncode != 0
    assert "RLMESH_ENV_ENDPOINT_TOKEN must not be empty" in result.stderr


def test_environment_token_is_enforced_and_reaches_run_loopback() -> None:
    result = _run_child(
        """
        server = rlmesh.EnvServer(Env(), "127.0.0.1:0")
        server.start()
        try:
            try:
                rlmesh.RemoteEnv(server.address)
            except ConnectionError as error:
                assert "Unauthenticated" in str(error), error
            else:
                raise AssertionError("a client without the token must be rejected")
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
