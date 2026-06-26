from __future__ import annotations

from typing import Any, cast, overload

from .._value_conversion import ValueBridge
from ..specs import SpaceSpec
from ..types import Value
from ._base import OutputT, Space
from ._internals import NATIVE_VALUE_BRIDGE
from .box import Box
from .dict import Dict
from .discrete import Discrete
from .multi_binary import MultiBinary
from .multi_discrete import MultiDiscrete
from .text import Text
from .tuple import Tuple

_SPACE_BY_KIND: dict[str, type[Space[Any]]] = {
    "box": Box,
    "discrete": Discrete,
    "multi_binary": MultiBinary,
    "multi_discrete": MultiDiscrete,
    "text": Text,
    "dict": Dict,
    "tuple": Tuple,
}


@overload
def space_from_spec(spec: SpaceSpec, *, bridge: None = None) -> Space[Value]: ...


@overload
def space_from_spec(spec: SpaceSpec, *, bridge: ValueBridge) -> Space[OutputT]: ...


def space_from_spec(
    spec: SpaceSpec,
    *,
    bridge: ValueBridge | None = None,
) -> Space[OutputT] | Space[Value]:
    """Create the named RLMesh space wrapper for a native spec.

    Args:
        spec: Native space specification.
        bridge: Optional backend value bridge for sample/contains values.

    Returns:
        Matching RLMesh space wrapper.
    """
    kind = spec.kind
    cls = _SPACE_BY_KIND.get(kind)
    if cls is None:
        raise ValueError(f"unsupported RLMesh space kind: {kind}")
    space = cast(Any, cls(spec))
    if bridge is None:
        return cast(Space[Value], space._with_bridge(NATIVE_VALUE_BRIDGE))
    return cast(Space[OutputT], space._with_bridge(bridge))
