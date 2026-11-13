#!/usr/bin/env python3
"""Gymnasium-facing client for the Unreal Reach Target EnvServer.

With the Unreal editor open and a static identity configured::

  python integrations/unreal/demos/reach_target/app.py --init-identity  # once
  python integrations/unreal/demos/reach_target/gym.py

Loads the Reach Target scene via demo control, then connects
``rlmesh.numpy.RemoteEnv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _DEMO_DIR.parent / "common"
for _p in (_COMMON_DIR, _DEMO_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gym_common import make_remote_env, smoke_remote_env

DEMO_ID = "reach_target"


def make_env(address: str | None = None, **remote_kwargs):
    """Connect to the serving Reach Target as ``rlmesh.numpy.RemoteEnv``."""
    return make_remote_env(address, demo_id=DEMO_ID, **remote_kwargs)


def main(argv: list[str] | None = None) -> int:
    del argv  # unused; kept for CLI symmetry
    return smoke_remote_env(demo_id=DEMO_ID)


if __name__ == "__main__":
    raise SystemExit(main())
