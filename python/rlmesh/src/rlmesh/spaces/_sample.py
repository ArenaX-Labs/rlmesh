from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from .._rlmesh import Tensor
from .._values import ValueBridge, identity_bridge
from ..specs import SpaceSpec
from ..types import Value
from ._base import SpaceBridge
from ._utils import EMPTY_SPACE_MAPPING, spec_details

_ARRAY_SAMPLE_KINDS = frozenset({"box", "multi_binary", "multi_discrete"})
_DTYPE_STRUCT_FORMATS = {
    "bool": "?",
    "uint8": "B",
    "int8": "b",
    "int16": "h",
    "int32": "i",
    "int64": "q",
    "uint16": "H",
    "uint32": "I",
    "uint64": "Q",
    "float16": "e",
    "float32": "f",
    "float64": "d",
}


def adapt_sample_tree(
    value: object,
    spec: SpaceSpec,
    array_leaf: Callable[[object, SpaceSpec], object],
) -> object:
    """Adapt tensor-like sample leaves in a possibly nested sample tree."""
    if spec.kind in _ARRAY_SAMPLE_KINDS:
        return array_leaf(value, spec)
    if spec.kind == "dict":
        values = cast(Mapping[str, object], value)
        details = spec_details(spec)
        raw_spaces = cast(
            Mapping[str, SpaceSpec], details.get("spaces", EMPTY_SPACE_MAPPING)
        )
        return {
            key: adapt_sample_tree(values[key], child, array_leaf)
            for key, child in raw_spaces.items()
        }
    if spec.kind == "tuple":
        values = cast(Sequence[object], value)
        details = spec_details(spec)
        raw_spaces = cast(list[SpaceSpec], details.get("spaces", []))
        return tuple(
            adapt_sample_tree(child_value, child_space, array_leaf)
            for child_value, child_space in zip(values, raw_spaces, strict=True)
        )
    return value


def native_sample(value: object, spec: SpaceSpec) -> Value:
    """Adapt native space samples to RLMesh Value leaves.

    The Rust sampler returns Python lists for tensor-like spaces. The native
    Python backend should expose those as RLMesh Tensor values, matching remote
    reset/step decoding when NumPy or Torch is not selected.
    """
    return cast(Value, adapt_sample_tree(value, spec, _tensor_from_sample))


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


def _tensor_from_sample(value: object, spec: SpaceSpec) -> Tensor:
    values = _flatten_sample(value)
    dtype = spec.dtype
    if dtype == "bfloat16":
        buffer = _pack_bfloat16(values)
    else:
        fmt = _DTYPE_STRUCT_FORMATS.get(dtype)
        if fmt is None:
            raise ValueError(f"unsupported tensor dtype {dtype!r}")
        buffer = struct.pack(f"<{len(values)}{fmt}", *values)
    return Tensor(buffer, spec.shape, dtype)


def _pack_bfloat16(values: Sequence[object]) -> bytes:
    """Pack floats as bfloat16; the struct module has no bfloat16 code.

    A bfloat16 is the top 16 bits of a float32, here with round-to-nearest-
    even applied to the dropped half.
    """
    out = bytearray()
    for value in values:
        (bits,) = struct.unpack("<I", struct.pack("<f", float(cast(float, value))))
        bits += 0x7FFF + ((bits >> 16) & 1)
        out += struct.pack("<H", (bits >> 16) & 0xFFFF)
    return bytes(out)


def _flatten_sample(value: object) -> list[object]:
    if isinstance(value, str | bytes | bytearray | memoryview):
        return [value]
    if isinstance(value, Mapping):
        raise TypeError("array-like sample leaves cannot be mappings")
    if isinstance(value, Sequence):
        values: list[object] = []
        items = cast(Sequence[object], value)
        for item in items:
            values.extend(_flatten_sample(item))
        return values
    return [value]
