"""Drive the native C++ Project Chrono env from a Python model.

The env (``chrono_reach_env``) is Chrono's 6-DOF industrial robot served
through the RLMesh C ABI. This model never sees the env's own layout: it
declares the inputs it wants once (``MODEL_SPEC``), and RLMesh resolves an
adapter against the tags the C++ env published -- resizing the camera, packing
the proprio vector, and mapping its action back onto the env's.

    uv run python examples/chrono/run_model.py 127.0.0.1:50051
    uv run python examples/chrono/run_model.py 127.0.0.1:50051 --policy random
    uv run python examples/chrono/run_model.py 127.0.0.1:50051 --view http:9000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import rlmesh.adapters as adapt
from rlmesh.numpy import Model, RemoteEnv

# The checkpoint's input format: a 224x224 camera, one proprio vector, the goal
# and the instruction under its own keys, and a normalized 3-D tool delta out.
MODEL_SPEC = adapt.ModelSpec(
    input={
        "image": adapt.Image(adapt.IMAGE_PRIMARY, size=224),
        "proprio": adapt.Concat(adapt.EEF_POS, adapt.JOINT_POS),
        "goal": adapt.Concat("x/target_pos"),
        "task": adapt.Text(adapt.INSTRUCTION),
    },
    output=adapt.Action(adapt.Actuator(adapt.ACTION_DELTA_POS, dim=3)),
)

# The gap (metres) at which the scripted policy's action saturates.
STEP_METRES = 0.03


def make_policy(
    kind: str, seed: int, pace: float
) -> Callable[[dict[str, Any]], np.ndarray]:
    """A predict function in MODEL_SPEC's format: scripted (seek the goal) or random."""
    rng = np.random.default_rng(seed)

    def predict(payload: dict[str, Any]) -> np.ndarray:
        assert payload["image"].shape == (224, 224, 3)
        if pace:
            time.sleep(pace)  # slow the loop down enough to watch
        if kind == "random":
            return rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        # Scripted: head straight for the goal (the proprio vector starts with
        # the tool position), easing off as the gap closes.
        eef = np.asarray(payload["proprio"][:3], dtype=np.float32)
        goal = np.asarray(payload["goal"], dtype=np.float32)
        return np.clip((goal - eef) / STEP_METRES, -1.0, 1.0).astype(np.float32)

    return predict


def main() -> None:
    """Connect, print the resolved adapter, run the episodes, report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("address", nargs="?", default="127.0.0.1:50051")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--policy", choices=["scripted", "random"], default="scripted")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--view",
        default=None,
        help='live viewer: "terminal" or "http:9000" (open http://localhost:9000)',
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=None,
        help="seconds to pause per step (default: 0.08 with --view, else 0)",
    )
    args = parser.parse_args()

    env = RemoteEnv(args.address)
    contract = env.env_contract
    print(f"connected to {contract.id} at {args.address}")
    print(f"  {contract.metadata.get('simulator')} / {contract.metadata.get('robot')}")
    print("\nresolved adapter (env tags x model spec):")
    print(adapt.resolve_from_contract(contract, MODEL_SPEC).explain())

    pace = args.pace if args.pace is not None else (0.08 if args.view else 0.0)
    model = Model(make_policy(args.policy, args.seed, pace), spec=MODEL_SPEC)
    seeds = [args.seed + i for i in range(args.episodes)]
    print(f"\nrunning {args.episodes} episodes, {args.policy} policy ...")
    if args.view:
        with model.session(env, view=args.view) as session:
            result = session.run(seeds=seeds)
            if sys.stdin.isatty():
                input("\nrun finished; press Enter to close the viewer ")
    else:
        result = model.run(env, seeds=seeds)

    for episode in result.episodes:
        print(
            f"  episode {episode.index}: seed={episode.seed} steps={episode.steps} "
            f"reward={episode.reward:.2f} success={episode.success}"
        )
    print(
        f"\nepisodes={result.num_episodes} steps={result.total_steps} "
        f"mean_reward={result.mean_reward:.2f} success_rate={result.success_rate}"
    )
    env.close()


if __name__ == "__main__":
    main()
