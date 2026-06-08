"""Run a sampled-action eval against any example environment server."""

import argparse

DEFAULT_ADDRESS = "127.0.0.1:5555"
DEFAULT_MAX_STEPS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one RLMesh environment endpoint."
    )
    parser.add_argument(
        "--address", default=DEFAULT_ADDRESS, help="environment endpoint address"
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from rlmesh.numpy import RemoteEnv

    env = RemoteEnv(args.address)

    print(f"connected to {env.address}")
    print(f"observation_space={env.observation_space!r}")
    print(f"action_space={env.action_space!r}")

    obs, info = env.reset(seed=0)
    for step in range(1, args.max_steps + 1):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"step={step} reward={reward:.3f}")
        if term or trunc:
            print("episode complete")
            break
    else:
        print(f"stopped after {args.max_steps} steps")

    env.close()


if __name__ == "__main__":
    main()
