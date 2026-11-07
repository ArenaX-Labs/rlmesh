"""0.1 options a session cannot honor raise at construction, naming the 0.2 form."""

from __future__ import annotations

import pytest
import rlmesh
from rlmesh import spaces


class Env:
    observation_space = spaces.Discrete(2)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: object = None):
        return 0, {}

    def step(self, action: object):
        return 0, 0.0, True, False, {}

    def close(self) -> None:
        pass


def test_session_clients_refuse_credentials_and_owner_shutdown() -> None:
    server = rlmesh.EnvServer(Env(), ":0/env")
    server.start()
    try:
        with pytest.raises(TypeError, match="unexpected keyword"):
            rlmesh.RemoteEnv(server.address, api_key="key")
        with pytest.raises(TypeError, match="unexpected keyword"):
            rlmesh.RemoteEnv(server.address, tls_ca_file="ca.pem")
        model = rlmesh.Model(lambda observation: 0)
        with pytest.raises(ValueError, match=r"server\.shutdown\(\)"):
            model.run(server.address, close_env=True)
        with pytest.raises(ValueError, match=r"server\.shutdown\(\)"):
            model.bind(server.address, close_env=True)
        with rlmesh.RemoteEnv(server.address) as remote:
            with pytest.raises(AttributeError, match="shutdown"):
                remote.shutdown()
    finally:
        server.shutdown()


def test_a_second_env_under_a_live_name_is_refused_at_construction() -> None:
    server = rlmesh.EnvServer(Env(), ":0/env")
    server.start()
    try:
        with pytest.raises(ConnectionError, match="participant env is already in"):
            rlmesh.EnvServer(Env(), server.address)
    finally:
        server.shutdown()
