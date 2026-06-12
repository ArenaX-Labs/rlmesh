"""Env registry: one module per environment, one line per entry below."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import rlmesh.adapters as adapt

from . import libero, simpler_bridge


@dataclass(frozen=True)
class EnvEntry:
    """Everything the eval harness needs to know about one environment."""

    spec: adapt.EnvIOSpec
    sample_obs: Callable[[], dict[str, Any]]


ENVS: dict[str, EnvEntry] = {
    "libero": EnvEntry(libero.SPEC, libero.sample_obs),
    "simpler-bridge": EnvEntry(simpler_bridge.SPEC, simpler_bridge.sample_obs),
}
