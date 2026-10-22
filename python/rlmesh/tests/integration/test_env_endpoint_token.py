"""A set env endpoint token fails serving: 0.2 sessions cannot check one yet."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("token", ["deployment-token", ""])
def test_env_server_refuses_a_set_endpoint_token(
    token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rlmesh
    from rlmesh import spaces

    class Env:
        observation_space = spaces.Discrete(2)
        action_space = spaces.Discrete(2)

        def reset(self, *, seed: int | None = None, options: object = None):
            return 0, {}

        def step(self, action: object):
            return 0, 0.0, True, False, {}

    monkeypatch.setenv("RLMESH_ENV_ENDPOINT_TOKEN", token)
    with pytest.raises(ValueError, match="safety step 6"):
        rlmesh.EnvServer(Env(), ":0/env")
