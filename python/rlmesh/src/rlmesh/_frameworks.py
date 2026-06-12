"""Internal scaffolding shared by framework-backed value modules.

Each framework module (``rlmesh.numpy``, ``rlmesh.torch``, ``rlmesh.jax``)
supplies three leaf callables and gets uniform tree-walking value
conversion; everything else in those modules is framework-specific
conversion code and the public client/model/sandbox classes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from ._rlmesh import Tensor
from ._values import decode_tree, encode_tree
from .types import Value

SUPPORTED_DTYPES: Final[tuple[str, ...]] = (
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)


class FrameworkBridge:
    """Tree-walking value conversion for one array framework.

    Implements the internal ``ValueAdapter`` protocol. Tensor leaves decode
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


__all__ = ["SUPPORTED_DTYPES", "FrameworkBridge"]
