#!/usr/bin/env python3
"""Gymnasium-facing client for the Unreal Chase Point EnvServer.

With the Unreal editor open::

  python integrations/unreal/demos/chase_point/gym.py

Loads the Chase Point scene via demo control, then connects.
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

DEMO_ID = "chase_point"


def make_env(address: str | None = None, **remote_kwargs):
    """Connect to the serving Chase Point env as ``rlmesh.numpy.RemoteEnv``."""
    return make_remote_env(address, demo_id=DEMO_ID, **remote_kwargs)


def main(argv: list[str] | None = None) -> int:
    del argv
    return smoke_remote_env(demo_id=DEMO_ID)


if __name__ == "__main__":
    raise SystemExit(main())
