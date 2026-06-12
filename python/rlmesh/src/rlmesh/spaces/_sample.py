from __future__ import annotations

from typing import Any, cast

from .._values import ValueBridge, identity_bridge
from ..specs import SpaceSpec
from ..types import Value
from ._base import SpaceBridge


def native_sample(value: object, spec: SpaceSpec) -> Value:
    """Adapt native space samples to RLMesh Value leaves.

    The Rust sampler emits RLMesh Tensor values for tensor-like spaces (and
    nested dicts/tuples thereof) directly, so the native Python backend exposes
    the sample tree unchanged, matching remote reset/step decoding when NumPy or
    Torch is not selected.
    """
    _ = spec
    return cast(Value, value)


def space_bridge_from_value_bridge(bridge: ValueBridge) -> SpaceBridge[Any]:
    """Build a space bridge from an existing runtime value bridge."""

    def sample(value: object, spec: SpaceSpec) -> object:
        return bridge.decode(native_sample(value, spec))

    def input(value: object, spec: SpaceSpec) -> object:
        _ = spec
        return bridge.encode(value)

    return SpaceBridge(sample, input)


NATIVE_SPACE_BRIDGE: SpaceBridge[Value] = cast(
    SpaceBridge[Value],
    space_bridge_from_value_bridge(identity_bridge),
)
