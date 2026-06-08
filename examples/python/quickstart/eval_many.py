"""Run one sampled-action evaluator against multiple RLMesh endpoints."""

import argparse
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MAX_STEPS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple RLMesh environment endpoints."
    )
    parser.add_argument("addresses", nargs="+", help="environment endpoint addresses")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    return parser.parse_args()


def evaluate(address: str, max_steps: int) -> str:
    from rlmesh.numpy import RemoteEnv

    env = RemoteEnv(address)
    try:
        lines = [
            f"{address}: connected",
            f"{address}: observation_space={env.observation_space!r}",
            f"{address}: action_space={env.action_space!r}",
        ]

        obs, info = env.reset(seed=0)
        for step in range(1, max_steps + 1):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            lines.append(f"{address}: step={step} reward={reward:.3f}")
            if term or trunc:
                lines.append(f"{address}: episode complete")
                break
        else:
            lines.append(f"{address}: stopped after {max_steps} steps")
        return "\n".join(lines)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    with ThreadPoolExecutor(max_workers=len(args.addresses)) as executor:
        futures = [
            executor.submit(evaluate, address, args.max_steps)
            for address in args.addresses
        ]
        for future in futures:
            print(future.result())


if __name__ == "__main__":
    main()
