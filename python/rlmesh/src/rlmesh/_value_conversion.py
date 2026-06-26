"""Internal value conversion shared by Python framework backends.

Framework modules (``rlmesh.numpy``, ``rlmesh.torch``, ``rlmesh.jax``) supply
leaf encoders/decoders for their tensor type. This module owns the shared
tree-walking logic for RLMesh value payloads so runtime, model, and adapter
paths agree on one canonical Python value shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final, Protocol, TypeVar, cast

from ._rlmesh import Tensor
from .specs import SpaceSpec
from .types import PrimitiveValue, Value

UNHANDLED: Final = object()
ValueT = TypeVar("ValueT")


class ValueBridge(Protocol):
    name: str

    def ensure_available(self) -> None: ...

    def decode(self, value: Value | None) -> object: ...

    def encode(self, value: object) -> Value: ...


class IdentityBridge:
    name: str = "rlmesh"

    def ensure_available(self) -> None:
        return None

    def decode(self, value: Value | None) -> Value | None:
        return value

    def encode(self, value: object) -> Value:
        return cast(Value, value)


def decode_tree(
    value: Value | None, leaf_decoder: Callable[[Tensor], ValueT]
) -> (
    ValueT
    | PrimitiveValue
    | list[object]
    | tuple[object, ...]
    | dict[str, object]
    | None
):
    if value is None:
        return None
    if isinstance(value, Tensor):
        return leaf_decoder(value)
    if isinstance(value, list):
        return [decode_tree(item, leaf_decoder) for item in value]
    if isinstance(value, tuple):
        return tuple(decode_tree(item, leaf_decoder) for item in value)
    if isinstance(value, dict):
        return {key: decode_tree(item, leaf_decoder) for key, item in value.items()}
    return value


def encode_tree(value: object, leaf_encoder: Callable[[object], object]) -> Value:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, dict):
        raw_mapping = cast(Mapping[object, object], value)
        return {
            str(key): encode_tree(item, leaf_encoder)
            for key, item in raw_mapping.items()
        }
    if isinstance(value, list):
        raw_items = cast(list[object], value)
        return [encode_tree(item, leaf_encoder) for item in raw_items]
    if isinstance(value, tuple):
        raw_items = cast(tuple[object, ...], value)
        return tuple(encode_tree(item, leaf_encoder) for item in raw_items)
    encoded = leaf_encoder(value)
    if encoded is UNHANDLED:
        return cast(Value, value)
    return cast(Value, encoded)


identity_bridge: ValueBridge = IdentityBridge()


def _bridge_or_identity(bridge: ValueBridge | None) -> ValueBridge:
    return identity_bridge if bridge is None else bridge


def to_value(value: object, bridge: ValueBridge | None = None) -> Value:
    """Encode a backend/native Python payload into the canonical Value tree."""
    return _bridge_or_identity(bridge).encode(value)


def from_value(value: Value | None, bridge: ValueBridge | None = None) -> object:
    """Decode a canonical Value tree into the requested backend."""
    return _bridge_or_identity(bridge).decode(value)


def encode_framework_array_batch(
    value: object,
    *,
    bridge: ValueBridge | None,
    space: SpaceSpec,
    num_envs: int,
) -> object:
    """Encode a single framework array batch into per-env Value leaves.

    Vector environment APIs commonly pass one array shaped
    ``(num_envs, *action_shape)`` for leaf array spaces. Splitting here keeps
    that backend policy with the value conversion system instead of each client
    guessing how to identify and encode framework batches.

    Non-array batches, composite spaces, text spaces, identity/native values, and
    count mismatches pass through unchanged so the native batched codec remains
    the final authority.
    """
    active_bridge = _bridge_or_identity(bridge)
    if active_bridge.name == identity_bridge.name:
        return value
    if space.kind in ("dict", "tuple", "text"):
        return value
    if not _is_framework_array_batch(value):
        return value
    try:
        per_env = list(cast(Iterable[object], value))
    except TypeError:
        return value
    if len(per_env) != num_envs:
        return value
    return [active_bridge.encode(item) for item in per_env]


def _is_framework_array_batch(value: object) -> bool:
    """Return true for a single framework array batch, not a container batch."""
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    if isinstance(value, (list, tuple)):
        return False
    if isinstance(value, Sequence):
        return False
    return hasattr(value, "__len__") and hasattr(value, "__iter__")


class FrameworkBridge:
    """Tree-walking value conversion for one array framework.

    Implements the internal ``ValueBridge`` protocol. Tensor leaves decode
    through ``decode_leaf``; arbitrary leaves encode through ``encode_leaf``,
    which returns ``UNHANDLED`` to pass a value through unchanged.
    Availability is checked once per ``decode``/``encode`` call.
    """

    def __init__(
        self,
        *,
        name: str,
        ensure_available: Callable[[], None],
        decode_leaf: Callable[[Tensor], object],
        encode_leaf: Callable[[object], object],
    ) -> None:
        self.name = name
        self._ensure_available = ensure_available
        self._decode_leaf = decode_leaf
        self._encode_leaf = encode_leaf

    def ensure_available(self) -> None:
        self._ensure_available()

    def decode(self, value: Value | None) -> object:
        self._ensure_available()
        return decode_tree(value, self._decode_leaf)

    def encode(self, value: object) -> Value:
        self._ensure_available()
        return encode_tree(value, self._encode_leaf)


__all__ = [
    "UNHANDLED",
    "FrameworkBridge",
    "ValueBridge",
    "encode_framework_array_batch",
    "from_value",
    "identity_bridge",
    "to_value",
]
