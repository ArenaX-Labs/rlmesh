"""Type-checker literals for space construction (editor IntelliSense only)."""

from __future__ import annotations

from typing import Literal, TypeAlias

FloatDType: TypeAlias = Literal["float16", "float32", "float64"]
IntDType: TypeAlias = Literal[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
]
BoolDType: TypeAlias = Literal["bool"]
DType: TypeAlias = FloatDType | IntDType | BoolDType

# ``object`` admits any third-party dtype without naming a framework; the Rust DType enum validates the name.
FloatDTypeLike: TypeAlias = FloatDType | object
IntDTypeLike: TypeAlias = IntDType | object
DiscreteDType: TypeAlias = Literal["int64", "int32", "uint8"]
DiscreteDTypeLike: TypeAlias = DiscreteDType | object

InfBound: TypeAlias = Literal["inf", "-inf"]
Bound: TypeAlias = float | InfBound
