"""Type-checker literals for space construction (editor IntelliSense only).

Runtime validation stays in Rust (``DType``); these just give autocomplete for
the dtype names. Keep them in sync with the Rust ``DType`` enum in
``crates/rlmesh-spaces/src/dtype.rs``.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

FloatDType: TypeAlias = Literal["float16", "float32", "float64"]
IntDType: TypeAlias = Literal[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"
]
BoolDType: TypeAlias = Literal["bool"]
DType: TypeAlias = FloatDType | IntDType | BoolDType

# Accepted dtype argument: our literals (so editors autocomplete the names) plus
# ``object`` so any third-party dtype -- ``np.float32``, ``torch.float32``, a jax
# dtype -- is allowed without naming a framework. ``_internals.dtype_name``
# coerces it to a name string and the Rust ``DType`` enum validates it.
FloatDTypeLike: TypeAlias = FloatDType | object
IntDTypeLike: TypeAlias = IntDType | object

# Infinite bounds: ``float(...)`` already parses these, so they work as Box
# bounds with no runtime change -- the literal just adds autocomplete.
InfBound: TypeAlias = Literal["inf", "-inf"]
Bound: TypeAlias = float | InfBound
