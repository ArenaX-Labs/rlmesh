"""Override registry: one module per special pairing, one line per entry below."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import rlmesh.adapters as adapt

from . import xvla_simpler_bridge

OVERRIDES: dict[tuple[str, str], Callable[[], adapt.AdapterBase[Any]]] = {
    ("xvla", "simpler-bridge"): xvla_simpler_bridge.XVLABridgeAdapter,
}
