"""Env registry: one module per environment, one line per entry below."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import rlmesh.adapters as adapt

from . import libero, simpler_bridge


@dataclass(frozen=True)
class EnvEntry:
    """Everything the eval harness needs to know about one environment."""

    annotations: adapt.EnvAnnotations
    observation_space: gym.spaces.Space[Any]
    action_space: gym.spaces.Space[Any]
    sample_obs: Callable[[], dict[str, Any]]


ENVS: dict[str, EnvEntry] = {
    "libero": EnvEntry(
        libero.ANNOTATIONS,
        libero.OBSERVATION_SPACE,
        libero.ACTION_SPACE,
        libero.sample_obs,
    ),
    "simpler-bridge": EnvEntry(
        simpler_bridge.ANNOTATIONS,
        simpler_bridge.OBSERVATION_SPACE,
        simpler_bridge.ACTION_SPACE,
        simpler_bridge.sample_obs,
    ),
}
