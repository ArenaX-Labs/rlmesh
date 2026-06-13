"""Model registry: one module per checkpoint, one line per entry below."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import rlmesh.adapters as adapt

from . import act, smolvla, xvla

# An escape-hatch adapter factory: given an env's tags and spaces,
# return a custom adapter (for models whose deployment needs stateful logic
# specs cannot express).
MakeAdapter = Callable[
    [adapt.EnvTags, "gym.spaces.Space[Any]", "gym.spaces.Space[Any]"],
    adapt.AdapterBase[Any],
]


@dataclass(frozen=True)
class ModelEntry:
    """Everything the eval harness needs to know about one checkpoint.

    ``make_adapter`` is the custom-adapter escape hatch: when set, the
    harness calls it with the env's tags and spaces instead of
    ``resolve()`` -- for models whose deployment needs stateful logic specs
    cannot express.
    """

    spec: adapt.ModelSpec
    load_predict_fn: Callable[[], Callable[[Mapping[str, Any]], Any]]
    make_adapter: MakeAdapter | None = None


MODELS: dict[str, ModelEntry] = {
    "act": ModelEntry(act.SPEC, act.load_predict_fn, act.make_adapter),
    "smolvla": ModelEntry(smolvla.SPEC, smolvla.load_predict_fn),
    "xvla": ModelEntry(xvla.SPEC, xvla.load_predict_fn),
}
