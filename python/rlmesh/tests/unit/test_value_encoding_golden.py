"""Cross-language byte contract (value-encoding-v1).

The little-endian bytes the RLMesh wire codec emits for a Box leaf must equal
what numpy produces for the same values — otherwise a tensor round-trips to
different numbers across the Rust and Python sides. This is the Python mirror of
the Rust golden in
``crates/rlmesh-spaces/src/scalar.rs::value_encoding_v1_float_golden`` and the
PyO3 packer regression
``python/rlmesh/rust/.../codec.rs::f16_pack_single_rounds_not_double_rounds``.

Keep all three in sync. This module needs only numpy (it pins the oracle), so it
runs without the native extension.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# 1.0 + 2^-11 (the f16 1.0 <-> 1.0009765625 midpoint) + 2^-25 (a hair above it,
# below f32 precision near 1.0). It rounds to f16 0x3C01 with a single f64->f16
# round; a double f64->f32->f16 round collapses it to 0x3C00. The sentinel that
# catches a regression to the old PyO3 `f16::from_f32(x as f32)` packing.
DOUBLE_ROUNDING = 1.0 + 1.0 / 2048.0 + 1.0 / 33_554_432.0

# (label, python value, numpy dtype, expected little-endian bytes)
GOLDEN: list[tuple[str, float, type, bytes]] = [
    ("f16 +0.0", 0.0, np.float16, b"\x00\x00"),
    ("f16 -0.0", -0.0, np.float16, b"\x00\x80"),
    ("f16 1.0", 1.0, np.float16, b"\x00\x3c"),
    ("f16 +inf", float("inf"), np.float16, b"\x00\x7c"),
    ("f16 -inf", float("-inf"), np.float16, b"\x00\xfc"),
    ("f16 max finite", 65504.0, np.float16, b"\xff\x7b"),
    ("f16 min subnormal", 5.9604644775390625e-8, np.float16, b"\x01\x00"),
    ("f16 double-rounding", DOUBLE_ROUNDING, np.float16, b"\x01\x3c"),
    ("f32 1.0", 1.0, np.float32, b"\x00\x00\x80\x3f"),
    ("f32 -0.0", -0.0, np.float32, b"\x00\x00\x00\x80"),
    ("f64 1.0", 1.0, np.float64, b"\x00\x00\x00\x00\x00\x00\xf0\x3f"),
    ("f64 -0.0", -0.0, np.float64, b"\x00\x00\x00\x00\x00\x00\x00\x80"),
]


@pytest.mark.parametrize(
    "label,value,dtype,expected", GOLDEN, ids=[row[0] for row in GOLDEN]
)
def test_numpy_matches_value_encoding_v1(label, value, dtype, expected):
    got = np.array(value, dtype=dtype).tobytes()
    assert got == expected, (
        f"{label}: numpy {dtype.__name__} produced {got.hex()}, want {expected.hex()}"
    )


def test_f16_quiet_nan_bytes_decode_to_nan():
    # The codec emits 0x7E00 for NaN; the exact payload is implementation-defined,
    # so only NaN-ness is contractual — numpy must read it back as NaN.
    value = np.frombuffer(b"\x00\x7e", dtype=np.float16)[0]
    assert np.isnan(value)


def test_host_is_little_endian():
    # The wire encoding is little-endian and numpy.frombuffer is native-endian, so
    # a big-endian host would silently byteswap. The package enforces this floor at
    # import; assert it here too so the contract is visible in the test suite.
    assert sys.byteorder == "little"
