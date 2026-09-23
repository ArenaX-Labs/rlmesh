"""A one-lane vector env serves as the scalar env it is."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import rlmesh
from rlmesh._single_lane import SingleLaneEnv, _info0
from rlmesh.numpy import Model, RemoteEnv


class _Success(gym.Wrapper):
    """CartPole that reports a task outcome through info, as a benchmark env would."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, "is_success": terminated}


def _one_lane(mode=gym.vector.AutoresetMode.NEXT_STEP):
    return gym.vector.SyncVectorEnv(
        [lambda: _Success(gym.make("CartPole-v1"))], autoreset_mode=mode
    )


def test_served_unbatched_over_the_wire():
    server = rlmesh.EnvServer(_one_lane(), "127.0.0.1:0")
    server.start()
    try:
        env = RemoteEnv(server.address)
        assert env.env_contract.num_envs == 1
        obs, _ = env.reset(seed=0)
        assert np.asarray(obs).shape == (4,)
        obs, reward, terminated, truncated, _ = env.step(0)
        assert np.asarray(obs).shape == (4,)
        assert isinstance(reward, float)
        env.close()
    finally:
        server.shutdown()


def test_run_gets_driver_owned_seeds_and_episode_cap():
    result = Model(lambda obs: 0).run(_one_lane(), seeds=[1, 2, 3], max_episode_steps=5)
    assert [episode.seed for episode in result.episodes] == [1, 2, 3]
    assert all(episode.steps <= 5 for episode in result.episodes)
    assert all(episode.success is not None for episode in result.episodes)


def test_info_takes_lane_zero_and_honors_masks():
    info = {
        "is_success": np.array([True]),
        "_is_success": np.array([True]),
        "hidden": np.array([7]),
        "_hidden": np.array([False]),
    }
    assert _info0(info) == {"is_success": True}


def test_same_step_is_refused():
    with pytest.raises(ValueError, match="SAME_STEP"):
        SingleLaneEnv(_one_lane(gym.vector.AutoresetMode.SAME_STEP))
