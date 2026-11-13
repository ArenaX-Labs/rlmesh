#!/usr/bin/env python3
"""Scripted Avoid Hazards client via ``rlmesh.run``.

  python integrations/unreal/demos/avoid_hazards/app.py --verbose
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
from policy import parse_obs, toward_goal_avoid


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

    if args.verbose:

        def policy_fn(obs: object):
            import numpy as np

            action = toward_goal_avoid(obs)
            goal, hazards = parse_obs(obs)
            nearest = min(float(np.linalg.norm(h)) for h in hazards)
            gdist = float(np.linalg.norm(goal))
            print(
                f"  goal_dist={gdist:.1f} nearest_hazard={nearest:.1f} "
                f"-> dir=({action[0]:+.2f}, {action[1]:+.2f})"
            )
            return action

        predict = policy_fn
    else:
        predict = toward_goal_avoid

    return run_scripted(
        predict,
        address=resolve_address(args.address) or None,
        demo_id="avoid_hazards",
        episodes=args.episodes,
        seeds=args.seeds,
        model_label="toward_goal_avoid (scripted, XY)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
