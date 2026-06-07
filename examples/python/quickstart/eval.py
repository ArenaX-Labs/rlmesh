"""Run a sampled-action eval against any example environment server."""

from __future__ import annotations

import rlmesh.numpy as rlmesh_numpy

ADDRESS = "127.0.0.1:5555"
MAX_STEPS = 64


def sample_action(env: rlmesh_numpy.RemoteEnv) -> object:
    """Sample one action from the remote server contract."""
    action_space = env.action_space
    if action_space is None:
        raise RuntimeError("remote environment did not advertise an action space")
    return action_space.sample()


def main() -> None:
    """Connect to whichever example server is running and step one episode."""
    env = rlmesh_numpy.RemoteEnv(ADDRESS)
    try:
        print(f"connected to {ADDRESS}")
        print(f"observation_space={env.observation_space!r}")
        print(f"action_space={env.action_space!r}")

        _observation, _info = env.reset(seed=0)
        for step in range(1, MAX_STEPS + 1):
            action = sample_action(env)
            _observation, reward, terminated, truncated, _info = env.step(action)
            print(f"step={step} reward={reward:.3f}")
            if terminated or truncated:
                print("episode complete")
                break
        else:
            print(f"stopped after {MAX_STEPS} steps")
    finally:
        env.close()


if __name__ == "__main__":
    main()
