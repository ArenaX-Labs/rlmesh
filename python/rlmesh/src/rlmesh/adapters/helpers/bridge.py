"""Encode values across the native adapter core boundary.

The native core speaks a small tagged-tuple encoding (see the ``adapters``
module of the ``rlmesh-python`` crate): arrays travel as
``("a", dtype, shape, bytes)`` (little-endian element bytes, matching the
repo-wide tensor/scalar codec), encoded PNG/JPEG images as ``("b", bytes)``
(decoded to RGB uint8 arrays on the native side), text as ``("t", str)``,
numbers as ``("n", float)``, lists as ``("l", [...])``, and nested mappings
as ``("m", {...})``. PIL image objects are converted to arrays here, since
holding one means PIL is already loaded in this process; nothing in this
module imports it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ...numpy import NumpyArray

# The element dtypes the native bridge accepts verbatim. Anything else is
# coerced to the nearest supported family (int64 / float32) before crossing,
# so the tagged-tuple codec only ever names a dtype the core can rebuild.
_SUPPORTED_DTYPES = ("uint8", "int32", "int64", "float32", "float64")


def _as_array(value: object) -> NumpyArray:
    import numpy as np

    if type(value).__module__.split(".")[0] == "PIL":
        converted = cast(Any, value).convert("RGB")
        return cast(NumpyArray, np.array(converted, dtype=np.uint8))
    array = cast(NumpyArray, np.asarray(value))
    if array.dtype == np.bool_:
        array = cast(NumpyArray, array.astype(np.uint8))
    if str(array.dtype) not in _SUPPORTED_DTYPES:
        target = np.int64 if array.dtype.kind in "iu" else np.float32
        array = cast(NumpyArray, array.astype(target))
    return cast(NumpyArray, np.ascontiguousarray(array))


def encode_value(value: object) -> tuple[Any, ...]:
    """Encode one raw value into the bridge form the native core accepts."""
    import numpy as np

    if isinstance(value, str):
        return ("t", value)
    if isinstance(value, bytes):
        return ("b", value)
    if isinstance(value, Mapping):
        return (
            "m",
            {
                str(key): encode_value(item)
                for key, item in cast(Mapping[Any, Any], value).items()
            },
        )
    if isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
        return ("n", float(cast(Any, value)))
    if isinstance(value, (list, tuple)):
        return ("l", [encode_value(item) for item in cast(Any, value)])
    array = _as_array(value)
    return ("a", str(array.dtype), tuple(array.shape), array.tobytes())


def decode_value(encoded: tuple[Any, ...]) -> Any:
    """Decode a bridge-encoded value produced by the native core."""
    import numpy as np

    tag = encoded[0]
    if tag == "a":
        _, dtype, shape, data = encoded
        # ``bytearray`` yields a writable buffer so the decoded array is
        # writable (a Gymnasium-style observation the caller can mutate).
        array = np.frombuffer(bytearray(data), dtype=dtype)
        return cast(NumpyArray, np.reshape(array, shape))
    if tag == "t" or tag == "n":
        return encoded[1]
    if tag == "l":
        return [decode_value(item) for item in encoded[1]]
    if tag == "m":
        return {key: decode_value(item) for key, item in encoded[1].items()}
    raise ValueError(f"unknown bridge value tag {tag!r}")


__all__ = ["decode_value", "encode_value"]
