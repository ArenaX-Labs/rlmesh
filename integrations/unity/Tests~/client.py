"""Drive the plain .NET host or the real Unity player through the Python API."""

import argparse
from pathlib import Path

import numpy as np
import rlmesh.numpy


def drive(address: str, episodes: int, *, counter: bool = False) -> None:
    """Assert observations, rewards and episode boundaries over the real wire."""
    with rlmesh.numpy.RemoteEnv(
        address, connect_timeout_seconds=10, request_timeout_seconds=10
    ) as env:
        for episode in range(episodes):
            observation, info = env.reset(seed=episode)
            if counter:
                assert observation == 0, observation
            else:
                assert np.asarray(observation).shape == (1,), observation
                assert info["env_id"] == "Unity-ReachTarget-v0", info
            for step in range(101):
                action = 1 if counter else (2 if observation[0] > 0 else 0)
                observation, reward, terminated, truncated, _ = env.step(action)
                if counter:
                    assert observation == step + 1, observation
                    assert reward == -(step + 1), reward
                else:
                    assert np.asarray(observation).shape == (1,), observation
                    assert np.isclose(reward, -abs(float(observation[0]))), reward
                if terminated or truncated:
                    if counter:
                        assert step == 2 and terminated and not truncated
                    break
            else:
                raise AssertionError("episode did not finish within 101 steps")
    print(f"Unity client OK: {episodes} episodes", flush=True)


def main() -> None:
    """The completion file is written only after the final response is received."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--counter", action="store_true")
    parser.add_argument("--complete-file", type=Path)
    args = parser.parse_args()
    drive(args.address, args.episodes, counter=args.counter)
    if args.complete_file:
        args.complete_file.write_text(str(args.episodes))


if __name__ == "__main__":
    main()
