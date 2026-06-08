from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, cast

from ..specs import SpaceSpec

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
