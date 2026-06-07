from __future__ import annotations

from typing import assert_type

import rlmesh
from rlmesh import types

options = rlmesh.ServeOptions(allow_remote_shutdown=True, idle_timeout_seconds=1.0)
assert_type(options.allow_remote_shutdown, bool)
assert_type(options.idle_timeout_seconds, float | None)

tensor = rlmesh.Tensor(b"\x00\x01", [2], "uint8")
assert_type(tensor, rlmesh.Tensor)
tensor_view = memoryview(tensor)
assert_type(tensor_view, memoryview)


class IntSpace:
    def sample(self) -> int:
        return 0

    def contains(self, value: object) -> bool:
        return isinstance(value, int)

    def seed(self, seed: int | None = None) -> int | None:
        return seed


class TinyEnv:
    observation_space = IntSpace()
    action_space = IntSpace()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: types.InfoDict | None = None,
    ) -> tuple[int, types.InfoDict]:
        return 0, {"seed": seed, "options": options}

    def step(self, action: int) -> tuple[int, float, bool, bool, types.InfoDict]:
        return action, 0.0, False, False, {}

    def close(self) -> None:
        return None


def accepts_env(env: types.EnvLike[int, int]) -> None:
    assert_type(env.reset(), tuple[int, types.InfoDict])


accepts_env(TinyEnv())
