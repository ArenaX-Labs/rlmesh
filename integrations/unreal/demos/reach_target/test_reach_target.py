#!/usr/bin/env python3
"""Drive an Unreal Reach Target env (thin wrapper around app.py).

Prefer::

  python integrations/unreal/demos/reach_target/app.py --init-identity   # once
  python integrations/unreal/demos/reach_target/app.py                   # each run
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parent
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from app import main as app_main


def main() -> int:
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
