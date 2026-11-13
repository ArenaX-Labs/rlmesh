#!/usr/bin/env python3
"""Evaluate saved SB3 PPO on live Unreal Collect Coins.

  python integrations/unreal/demos/collect_coins/eval.py
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
    DEFAULT_EVAL_EPISODES,
    default_model_path,
    eval_ppo,
    load_suite_dotenv,
    resolve_address,
)

ARTIFACTS_DIR = _DEMO_DIR / "artifacts"


def main(argv: list[str] | None = None) -> int:
    load_suite_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--address", default="")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        returns = eval_ppo(
            address=resolve_address(args.address) or None,
            model_path=args.model or default_model_path(ARTIFACTS_DIR),
            artifacts_dir=ARTIFACTS_DIR,
            episodes=args.episodes,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: eval failed: {exc}", file=sys.stderr)
        return 1
    print(f"collect_coins eval finished successfully (returns={returns}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
