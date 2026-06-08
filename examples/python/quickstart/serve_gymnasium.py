"""Serve a Gymnasium environment through RLMesh."""

import argparse

DEFAULT_ADDRESS = "127.0.0.1:5555"
DEFAULT_ENV_ID = "CartPole-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a Gymnasium environment through RLMesh."
    )
    parser.add_argument(
        "--env-id", default=DEFAULT_ENV_ID, help="Gymnasium environment id"
    )
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="bind address")
    parser.add_argument(
        "--render-mode", default=None, help="optional Gymnasium render mode"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import gymnasium as gym
    from rlmesh import EnvServer

    make_kwargs = {}
    if args.render_mode is not None:
        make_kwargs["render_mode"] = args.render_mode

    env = gym.make(args.env_id, **make_kwargs)
    server = EnvServer(env, args.address)
    print(f"serving {args.env_id} on {server.address}")
    print(f"observation_space={env.observation_space!r}")
    print(f"action_space={env.action_space!r}")
    server.serve()


if __name__ == "__main__":
    main()
