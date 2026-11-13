#!/usr/bin/env python3
"""Gymnasium-facing client for the Unreal Reach Target EnvServer.

With the Unreal editor Serving Reach Target::

  python integrations/unreal/demos/reach_target/app.py --init-identity  # once
  # Press Play with an RLMesh Reach Target component
  python integrations/unreal/demos/reach_target/gym.py

Connects ``rlmesh.numpy.RemoteEnv`` using ``RLMESH_UNREAL_ADDR`` / ``--address``.
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


def make_env(address: str | None = None, **remote_kwargs):
    """Connect to the serving Reach Target as ``rlmesh.numpy.RemoteEnv``."""
    return make_remote_env(address, **remote_kwargs)


def main(argv: list[str] | None = None) -> int:
    del argv  # unused; kept for CLI symmetry
    return smoke_remote_env()


if __name__ == "__main__":
    raise SystemExit(main())
