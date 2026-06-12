from __future__ import annotations

from typing import Any, cast

from .._values import ValueBridge, identity_bridge
from ..specs import SpaceSpec
from ..types import Value
from ._base import SpaceBridge


def space_bridge_from_value_bridge(bridge: ValueBridge) -> SpaceBridge[Any]:
    """Build a space bridge from an existing runtime value bridge.

    The Rust sampler already emits RLMesh ``Value`` trees (Tensor leaves for
    tensor-like spaces, nested dicts/tuples thereof), so sampling only routes
    the native value through the framework bridge's ``decode``; the spec is not
    needed to reshape it.
    """

    def sample(value: object, _spec: SpaceSpec) -> object:
        return bridge.decode(cast(Value, value))

    def input(value: object, _spec: SpaceSpec) -> object:
        return bridge.encode(value)

    return SpaceBridge(sample, input)


NATIVE_SPACE_BRIDGE: SpaceBridge[Value] = cast(
    SpaceBridge[Value],
    space_bridge_from_value_bridge(identity_bridge),
)
