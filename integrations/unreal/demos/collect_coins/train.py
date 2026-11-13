#!/usr/bin/env python3
"""Train SB3 PPO on live Unreal Collect Coins.

  pip install -r integrations/unreal/demos/common/requirements-gym-suite.txt
  python integrations/unreal/demos/collect_coins/train.py
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

ARTIFACTS_DIR = _DEMO_DIR / "artifacts"


def main(argv: list[str] | None = None) -> int:
    load_suite_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TRAIN_TIMESTEPS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        path = train_ppo(
            address=resolve_address(args.address) or None,
            total_timesteps=args.timesteps,
            model_path=args.out or default_model_path(ARTIFACTS_DIR),
            artifacts_dir=ARTIFACTS_DIR,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: train failed: {exc}", file=sys.stderr)
        return 1
    print(f"collect_coins train finished successfully ({path}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
