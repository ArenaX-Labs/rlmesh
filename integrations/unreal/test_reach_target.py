#!/usr/bin/env python3
"""Drive an Unreal Reach Target env (thin wrapper around reach_target_app).

Prefer::

  python integrations/unreal/reach_target_app.py --init-identity   # once
  python integrations/unreal/reach_target_app.py                   # each run

This script keeps the older CLI shape and delegates to the same policy / .env
resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path

_UNREAL_DIR = Path(__file__).resolve().parent
if str(_UNREAL_DIR) not in sys.path:
    sys.path.insert(0, str(_UNREAL_DIR))

from reach_target_app import main as app_main


def main() -> int:
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
