#!/usr/bin/env python3
"""Scripted Chase Point client (toward_goal) via ``rlmesh.run``.

With the Unreal editor open (same host identity / .env as Reach Target)::

  python integrations/unreal/demos/chase_point/app.py --verbose

Loads the Chase Point scene automatically, then connects.
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
from policy import gap_xy, toward_goal


def main(argv: list[str] | None = None) -> int:
    load_suite_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--address",
        default="",
        help="Full Serving address override (else RLMESH_UNREAL_ADDR from .env)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.environ.get(EPISODES_VAR, "1") or "1"),
        help="Episodes when --seeds is not set",
    )
    parser.add_argument(
        "--seeds",
        default="",
        help="Comma-separated episode seeds (overrides --episodes)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print XY gap and direction each step",
    )
    args = parser.parse_args(argv)

    if args.verbose:

        def policy_fn(obs: object):
            import numpy as np

            action = toward_goal(obs)
            gap = gap_xy(obs)
            dist = float(np.linalg.norm(gap))
            print(
                f"  gap=({gap[0]:+.1f}, {gap[1]:+.1f}) dist={dist:.1f} "
                f"-> dir=({action[0]:+.2f}, {action[1]:+.2f})"
            )
            return action

        predict = policy_fn
    else:
        predict = toward_goal

    return run_scripted(
        predict,
        address=resolve_address(args.address) or None,
        episodes=args.episodes,
        seeds=args.seeds,
        model_label="toward_goal (scripted, XY)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
