"""Run a tiny local model/env loop and print per-episode seed context."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class TwoStepEnv:
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(0, 10, shape=(1,), dtype=np.int64)
        self.action_space = gym.spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _ = options
        self._step = 0
        print(f"env reset seed={seed}")
        return np.array([0], dtype=np.int64), {}

    def step(self, action: object):
        _ = action
        self._step += 1
        terminated = self._step >= 2
        return np.array([self._step], dtype=np.int64), 1.0, terminated, False, {}

    def close(self) -> None:
        return None


def predict(observation, context):
    print(
        "model predict "
        f"observation={observation.tolist()} "
        f"episode_id={context['episode_id'] or '<unset>'} "
        f"episode_seed={context['episode_seed']}"
    )
    return 0


def main() -> None:
    from rlmesh.numpy import Model

    result = Model(predict).run(TwoStepEnv(), seeds=[7, 8], max_episodes=2)
    print("episode report seeds:", [episode.seed for episode in result.episodes])


if __name__ == "__main__":
    main()
