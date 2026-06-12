"""Internal helpers for the adapters package: arrays and the native bridge."""

from .arrays import NumpyArray, ensure_available
from .bridge import decode_value, encode_value

__all__ = [
    "NumpyArray",
    "decode_value",
    "encode_value",
    "ensure_available",
]
