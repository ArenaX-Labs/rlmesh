"""Internal value conversion helpers for Python-facing SDK modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, Protocol, TypeVar, cast

from ._rlmesh import Tensor
from .types import PrimitiveValue, Value

UNHANDLED: Final = object()
ValueT = TypeVar("ValueT")


class ValueAdapter(Protocol):
    name: str

    def ensure_available(self) -> None: ...

    def decode(self, value: Value | None) -> object: ...

    def encode(self, value: object) -> Value: ...


class IdentityAdapter:
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


identity_adapter: ValueAdapter = IdentityAdapter()

__all__ = [
    "UNHANDLED",
    "IdentityAdapter",
    "ValueAdapter",
    "decode_tree",
    "encode_tree",
    "identity_adapter",
]
