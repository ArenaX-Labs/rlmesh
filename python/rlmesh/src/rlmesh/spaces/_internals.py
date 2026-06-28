"""Internal space helpers: spec accessors plus the native-value bridge."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

from .._value_conversion import ValueBridge, identity_bridge
from ..specs import SpaceSpec

# The native value bridge: spaces sample/contain RLMesh ``Value`` trees as-is.
NATIVE_VALUE_BRIDGE: ValueBridge = identity_bridge

EMPTY_SPACE_MAPPING: Mapping[str, object] = MappingProxyType({})


def space_shape(value: object) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(dim) for dim in cast(Sequence[Any], value)]


def shape(shape: Sequence[int] | None) -> list[int]:
    if shape is None:
        raise TypeError("shape is required")
    return [int(dim) for dim in shape]


def require_float(value: float | str | None, name: str) -> float:
    if value is None:
        raise TypeError(f"{name} is required")
    # float() accepts 'inf'/'-inf' (valid Box bounds) but also 'nan'; a NaN bound
    # makes every contains()/clamp comparison False, so reject it at the boundary.
    result = float(value)
    if math.isnan(result):
        raise ValueError(f"{name} must be a real number, not NaN")
    return result


def dtype_name(dtype: object) -> str:
    """Coerce any dtype-like to a name string for the Rust ``DType`` validator.

    Framework-agnostic, importing nothing: numpy dtype objects expose ``.name``,
    numpy/python scalar *types* expose ``__name__``, and ``torch``/``jax`` dtypes
    stringify as ``"<framework>.<name>"`` so the last segment is the name. The
    name is validated against our actual dtypes by the space constructor.
    """
    if isinstance(dtype, str):
        return dtype
    # numpy-fluent users write dtype=float / int (numpy reads these as float64 /
    # int64). The __name__ probe below would yield 'float'/'int', which the Rust
    # DType validator rejects, so map the builtins explicitly. ponytail: int ->
    # int64 is rlmesh's canonical default (numpy's int_ is int32 on Windows); pass
    # an explicit np.dtype or 'int32' for the platform-native width. (bool already
    # resolves: bool.__name__ == 'bool' is a valid dtype name.)
    builtin = {float: "float64", int: "int64"}.get(dtype)  # type: ignore[arg-type]
    if builtin is not None:
        return builtin
    # ponytail: attribute probe, not a per-framework table; the str() fallback
    # is best-effort and Rust rejects anything that isn't a real dtype name.
    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None)
    return name if isinstance(name, str) else str(dtype).rsplit(".", 1)[-1]


def spec_details(spec: SpaceSpec) -> Mapping[str, object]:
    return cast(Mapping[str, object], spec._details())


def spec_to_dict(spec: SpaceSpec) -> dict[str, object]:
    return spec._to_dict()
