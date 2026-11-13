#!/usr/bin/env python3
"""Train SB3 PPO on the live Unreal Chase Point env.

With the Unreal editor open and suite deps::

  pip install -r integrations/unreal/demos/common/requirements-gym-suite.txt
  python integrations/unreal/demos/chase_point/train.py

Loads Chase Point via demo control, then trains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _DEMO_DIR.parent / "common"
for _p in (_COMMON_DIR, _DEMO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gym_common import (
    DEFAULT_TRAIN_TIMESTEPS,
    default_model_path,
    load_suite_dotenv,
    resolve_address,
    train_ppo,
)

DEMO_ID = "chase_point"
ARTIFACTS_DIR = _DEMO_DIR / "artifacts"


def main(argv: list[str] | None = None) -> int:
    load_suite_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="", help="Serving address override")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TRAIN_TIMESTEPS,
        help=f"PPO total_timesteps (default {DEFAULT_TRAIN_TIMESTEPS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (default demos/chase_point/artifacts/ppo.zip)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        path = train_ppo(
            demo_id=DEMO_ID,
            address=resolve_address(args.address) or None,
            total_timesteps=args.timesteps,
            model_path=args.out or default_model_path(ARTIFACTS_DIR),
            artifacts_dir=ARTIFACTS_DIR,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: train failed: {exc}", file=sys.stderr)
        return 1
    print(f"chase_point train finished successfully ({path}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
