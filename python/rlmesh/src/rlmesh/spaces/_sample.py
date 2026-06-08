from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from .._rlmesh import Tensor
from .._values import ValueAdapter, identity_adapter
from ..specs import SpaceSpec
from ..types import Value
from ._base import SpaceAdapter
from ._utils import EMPTY_SPACE_MAPPING, spec_details

_ARRAY_SAMPLE_KINDS = frozenset({"box", "multi_binary", "multi_discrete"})
_DTYPE_STRUCT_FORMATS = {
    "bool": "?",
    "uint8": "B",
    "int32": "i",
    "int64": "q",
    "float16": "e",
    "float32": "f",
    "float64": "d",
}


def adapt_sample_tree(
    value: object,
    spec: SpaceSpec,
    array_adapter: Callable[[object, SpaceSpec], object],
) -> object:
    """Adapt tensor-like sample leaves in a possibly nested sample tree."""
    if spec.kind in _ARRAY_SAMPLE_KINDS:
        return array_adapter(value, spec)
    if spec.kind == "dict":
        values = cast(Mapping[str, object], value)
        details = spec_details(spec)
        raw_spaces = cast(
            Mapping[str, SpaceSpec], details.get("spaces", EMPTY_SPACE_MAPPING)
        )
        return {
            key: adapt_sample_tree(values[key], child, array_adapter)
            for key, child in raw_spaces.items()
        }
    if spec.kind == "tuple":
        values = cast(Sequence[object], value)
        details = spec_details(spec)
        raw_spaces = cast(list[SpaceSpec], details.get("spaces", []))
        return tuple(
            adapt_sample_tree(child_value, child_space, array_adapter)
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


def space_adapter_from_value_adapter(adapter: ValueAdapter) -> SpaceAdapter[Any]:
    """Build a space adapter from an existing runtime value adapter."""

    def sample(value: object, spec: SpaceSpec) -> object:
        return adapter.decode(native_sample(value, spec))

    def input(value: object, spec: SpaceSpec) -> object:
        _ = spec
        return adapter.encode(value)

    return SpaceAdapter(sample, input)


NATIVE_SPACE_ADAPTER: SpaceAdapter[Value] = cast(
    SpaceAdapter[Value],
    space_adapter_from_value_adapter(identity_adapter),
)


def _tensor_from_sample(value: object, spec: SpaceSpec) -> Tensor:
    values = _flatten_sample(value)
    dtype = spec.dtype
    fmt = _DTYPE_STRUCT_FORMATS.get(dtype)
    if fmt is None:
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    buffer = struct.pack(f"<{len(values)}{fmt}", *values)
    return Tensor(buffer, spec.shape, dtype)


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
