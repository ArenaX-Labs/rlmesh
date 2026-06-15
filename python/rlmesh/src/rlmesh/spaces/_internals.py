"""Internal space helpers: spec accessors plus the native-value space bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from .._framework_bridge import ValueBridge, identity_bridge
from ..specs import SpaceSpec
from ..types import Value

if TYPE_CHECKING:
    from ._base import SpaceBridge

EMPTY_SPACE_MAPPING: Mapping[str, object] = MappingProxyType({})


def space_shape(value: object) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(dim) for dim in cast(Sequence[Any], value)]


def shape(shape: Sequence[int] | None) -> list[int]:
    if shape is None:
        raise TypeError("shape is required")
    return [int(dim) for dim in shape]


def require_float(value: float | None, name: str) -> float:
    if value is None:
        raise TypeError(f"{name} is required")
    return float(value)


def spec_details(spec: SpaceSpec) -> Mapping[str, object]:
    return cast(Mapping[str, object], spec._details())


def spec_to_dict(spec: SpaceSpec) -> dict[str, object]:
    return spec._to_dict()


def space_bridge_from_value_bridge(bridge: ValueBridge) -> SpaceBridge[Any]:
    """Build a space bridge from an existing runtime value bridge.

    The Rust sampler already emits RLMesh ``Value`` trees (Tensor leaves for
    tensor-like spaces, nested dicts/tuples thereof), so sampling only routes
    the native value through the framework bridge's ``decode``; the spec is not
    needed to reshape it.
    """
    from ._base import SpaceBridge

    def sample(value: object, _spec: SpaceSpec) -> object:
        return bridge.decode(cast(Value, value))

    def input(value: object, _spec: SpaceSpec) -> object:
        return bridge.encode(value)

    return SpaceBridge(sample, input)


def __getattr__(name: str) -> object:
    # NATIVE_SPACE_BRIDGE is built lazily: it needs ``_base.SpaceBridge``, but
    # ``_base`` imports this module for its spec helpers, so eager construction at
    # import time would race that partial import. Built (and cached) on first access.
    if name == "NATIVE_SPACE_BRIDGE":
        bridge = cast(
            "SpaceBridge[Value]",
            space_bridge_from_value_bridge(identity_bridge),
        )
        globals()["NATIVE_SPACE_BRIDGE"] = bridge
        return bridge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
