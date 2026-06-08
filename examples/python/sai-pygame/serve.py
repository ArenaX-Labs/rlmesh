"""Serve the SAI Pygame SquidHunt environment through RLMesh."""

import argparse

DEFAULT_ADDRESS = "127.0.0.1:5555"
ENV_ID = "sai_pygame:SquidHunt-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Serve {ENV_ID} through RLMesh.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import gymnasium as gym
    from rlmesh import EnvServer

    env = gym.make(ENV_ID)
    server = EnvServer(env, args.address)
    server.start()
    print(f"serving {ENV_ID} on {server.address}")
    print(f"observation_space={env.observation_space!r}")
    print(f"action_space={env.action_space!r}")
    server.wait()


if __name__ == "__main__":
    main()
