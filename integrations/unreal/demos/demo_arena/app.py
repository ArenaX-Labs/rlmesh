#!/usr/bin/env python3
"""Scripted Demo Arena client via ``rlmesh.run``.

  python integrations/unreal/demos/demo_arena/app.py --verbose
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _DEMO_DIR.parent / "common"
for _p in (_COMMON_DIR, _DEMO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gym_common import load_suite_dotenv, resolve_address, run_scripted
from paths import EPISODES_VAR
from policy import nearest_target_gap, parse_obs, reset_arena_policy_state, toward_goal_arena


def main(argv: list[str] | None = None) -> int:
    load_suite_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="")
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get(EPISODES_VAR, "1") or "1"),
    )
    parser.add_argument("--seeds", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    reset_arena_policy_state()

    if args.verbose:

        def policy_fn(obs: object):
            import numpy as np

            action = toward_goal_arena(obs)
            goal, coins, obstacles, walls = parse_obs(obs)
            _tgt, tdist = nearest_target_gap(goal, coins)
            coins_left = sum(1 for c in coins if float(np.linalg.norm(c)) > 1e-6)
            goal_left = 1 if float(np.linalg.norm(goal)) > 1e-6 else 0
            nearest_obs = min(
                (float(np.linalg.norm(o)) for o in obstacles),
                default=float("inf"),
            )
            nearest_wall = float(np.min(walls)) if walls.size else float("inf")
            print(
                f"  target={tdist:.1f} coins_left={coins_left} goal_left={goal_left} "
                f"obstacle={nearest_obs:.1f} wall={nearest_wall:.1f} "
                f"-> dir=({action[0]:+.2f}, {action[1]:+.2f})"
            )
            return action

        predict = policy_fn
    else:
        predict = toward_goal_arena

    return run_scripted(
        predict,
        address=resolve_address(args.address) or None,
        demo_id="demo_arena",
        episodes=args.episodes,
        seeds=args.seeds,
        model_label="coins then goal (scripted XY)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
