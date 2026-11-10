#!/usr/bin/env python3
"""Gymnasium-facing client for the Unreal Reach Target EnvServer (no second server).

With Unreal in Play and a static identity configured::

  python integrations/unreal/reach_target_app.py --init-identity  # once
  # Play in Unreal, then:
  python -c "from reach_target_gym import make_env; e=make_env(); print(e.reset()); e.close()"

This is a thin helper around ``rlmesh.numpy.RemoteEnv``. The PoC claim is
**RLMesh env in Unreal (Gym-style API)** — not a registered Gymnasium env id.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_UNREAL_DIR = Path(__file__).resolve().parent
if str(_UNREAL_DIR) not in sys.path:
    sys.path.insert(0, str(_UNREAL_DIR))

from reach_target_dotenv import apply_dotenv
from reach_target_paths import DOTENV_PATH, UNREAL_ADDR_VAR


def make_env(address: str | None = None, **remote_kwargs):
    """Connect to the serving Reach Target as ``rlmesh.numpy.RemoteEnv``.

    When gymnasium is installed, the client is a ``gymnasium.Env``
    (``reset`` / ``step`` / spaces). Pass ``address`` or set
    ``RLMESH_UNREAL_ADDR`` / ``integrations/unreal/.env``.
    """
    apply_dotenv(DOTENV_PATH)
    addr = (address or os.environ.get(UNREAL_ADDR_VAR, "")).strip()
    if not addr:
        raise ValueError(
            "No Serving address. Run reach_target_app.py --init-identity, "
            "press Play in Unreal, or pass address=..."
        )
    from rlmesh.numpy import RemoteEnv

    return RemoteEnv(addr, **remote_kwargs)


def main(argv: list[str] | None = None) -> int:
    del argv  # unused; kept for CLI symmetry
    try:
        env = make_env()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        obs, info = env.reset()
        print(f"reset obs={obs} info_keys={sorted(info)}")
        print(f"observation_space={env.observation_space}")
        print(f"action_space={env.action_space}")
        action = env.action_space.sample()
        step = env.step(action)
        print(f"step action={action} -> obs/reward/term/trunc/info len={len(step)}")
    finally:
        env.close()
    print("reach_target_gym.py finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
