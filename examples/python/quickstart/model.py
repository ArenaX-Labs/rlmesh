"""Run a tiny RLMesh model worker against an environment endpoint."""

import argparse

DEFAULT_ADDRESS = "127.0.0.1:5555"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny model against an RLMesh endpoint."
    )
    parser.add_argument(
        "--address", default=DEFAULT_ADDRESS, help="environment endpoint address"
    )
    parser.add_argument("--episodes", type=int, default=1, help="episodes to run")
    return parser.parse_args()


def predict(observation):
    return 0


def main() -> None:
    args = parse_args()

    from rlmesh.numpy import Model

    model = Model(predict)
    model.run(args.address, max_episodes=args.episodes)


if __name__ == "__main__":
    main()
