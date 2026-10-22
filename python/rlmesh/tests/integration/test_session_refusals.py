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


@pytest.mark.parametrize("token", ["deployment-token", ""])
def test_env_server_refuses_a_set_endpoint_token(
    token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RLMESH_ENV_ENDPOINT_TOKEN", token)
    with pytest.raises(ValueError, match="safety step 6"):
        rlmesh.EnvServer(Env(), ":0/env")


@pytest.mark.parametrize(
    ("option", "why"),
    [
        ({"allow_remote_shutdown": True}, r"server\.shutdown\(\)"),
        ({"idle_timeout_seconds": 5.0}, "no idle shutdown"),
        ({"drain_timeout_seconds": 5.0}, "nothing to drain"),
        ({"compress_responses": True}, "not compressed"),
    ],
)
def test_session_servers_refuse_v1_serve_options(option: dict, why: str) -> None:
    options = rlmesh.ServeOptions(**option)
    with pytest.raises(ValueError, match=why):
        rlmesh.EnvServer(Env(), ":0/env", options=options)
    with pytest.raises(ValueError, match=why):
        rlmesh.Model(lambda observation: 0).serve(":0", name="m", options=options)


def test_session_clients_refuse_credentials_and_owner_shutdown() -> None:
    server = rlmesh.EnvServer(Env(), ":0/env")
    server.start()
    try:
        with pytest.raises(ValueError, match="safety steps 6 and 9"):
            rlmesh.RemoteEnv(server.address, api_key="key")
        with pytest.raises(ValueError, match="safety steps 6 and 9"):
            rlmesh.RemoteEnv(server.address, tls_ca_file="ca.pem")
        model = rlmesh.Model(lambda observation: 0)
        with pytest.raises(ValueError, match=r"server\.shutdown\(\)"):
            model.run(server.address, close_env=True)
        with pytest.raises(ValueError, match=r"server\.shutdown\(\)"):
            model.session(server.address, close_env=True)
        with rlmesh.RemoteEnv(server.address) as remote:
            with pytest.raises(TypeError, match=r"server\.shutdown\(\)"):
                remote.shutdown()
    finally:
        server.shutdown()
