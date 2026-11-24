from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
import pytest


class LaneEnv:
    def __init__(self, length: int, step_seconds: float = 0.0) -> None:
        from rlmesh import spaces

        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,), dtype="float32")
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype="float32")
        self.length = length
        self.step_seconds = step_seconds
        self.step_count = 0
        self.reset_seeds: list[int | None] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, object] | None = None
    ) -> tuple[Any, dict[str, object]]:
        _ = options
        self.reset_seeds.append(seed)
        self.step_count = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, object]]:
        _ = action
        if self.step_seconds:
            time.sleep(self.step_seconds)
        self.step_count += 1
        done = self.step_count >= self.length
        return np.zeros(2, dtype=np.float32), 1.0, done, False, {}

    def close(self) -> None:
        return None


def test_env_server_hosts_a_list_of_envs_as_independently_reset_lanes() -> None:
    import rlmesh
    import rlmesh.numpy as rlmesh_numpy

    lanes = [LaneEnv(1), LaneEnv(3, step_seconds=0.1)]
    try:
        server = rlmesh.EnvServer(cast("Any", lanes), ":0/env")
    except ConnectionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip("local tcp bind is not permitted in this environment")
        raise
    server.start()
    assert server.env_contract.num_envs == 2

    seen: list[Any] = []

    def predict(observation: Any) -> Any:
        seen.append(np.asarray(observation))
        return np.zeros(2, dtype=np.float32)

    try:
        result = rlmesh_numpy.Model(predict)._run_local_for_episodes(
            server.address, episodes=4, seeds=[11, 12, 13, 14]
        )
    finally:
        server.shutdown()

    assert seen and seen[0].shape == (2,)
    assert len(result["episodes"]) == 4
    assert lanes[0].reset_seeds == [11, 13, 14]
    assert lanes[1].reset_seeds == [12]
    assert sorted(seed for lane in lanes for seed in lane.reset_seeds) == [
        11,
        12,
        13,
        14,
    ]
