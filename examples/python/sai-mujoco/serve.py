"""Serve the SAI MuJoCo color-sort environment through RLMesh."""

import argparse

DEFAULT_ADDRESS = "127.0.0.1:5555"
ENV_ID = "sai_mujoco:So101IkColorSortPickPlace-v0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Serve {ENV_ID} through RLMesh.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    parser.add_argument(
        "--render-mode", default="rgb_array", help="Gymnasium render mode"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import gymnasium as gym
    from rlmesh import EnvServer

    env = gym.make(ENV_ID, render_mode=args.render_mode)
    server = EnvServer(env, args.address)
    server.start()
    print(f"serving {ENV_ID} on {server.address}")
    print(f"observation_space={env.observation_space!r}")
    print(f"action_space={env.action_space!r}")
    server.wait()


if __name__ == "__main__":
    main()
