from __future__ import annotations

from typing import assert_type

import rlmesh
from rlmesh import spaces
from rlmesh.specs import EnvContract
from rlmesh.types import InfoDict, Value


class TinyEnv:
    observation_space = spaces.Discrete(2)
    action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: InfoDict | None = None,
    ) -> tuple[int, InfoDict]:
        return 0, {"seed": seed, "options": options}

    def step(self, action: int) -> tuple[int, float, bool, bool, InfoDict]:
        return action, 1.0, True, False, {}

    def close(self) -> None:
        return None


server = rlmesh.EnvServer(
    TinyEnv(),
    host="127.0.0.1",
    port=0,
    options=rlmesh.ServeOptions(allow_remote_shutdown=True),
)
assert_type(server.address, str)
assert_type(server.env_contract, EnvContract)
assert_type(server.spec, EnvContract)
assert_type(server.wait(0), bool)
assert_type(server.wait(timeout=0), bool)


def predict(observation: Value) -> Value:
    return observation


model = rlmesh.Model(predict)
assert_type(model, rlmesh.Model)
