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


INT_GOLDEN: list[tuple[str, int, type, bytes]] = [
    ("i8 0", 0, np.int8, b"\x00"),
    ("i8 1", 1, np.int8, b"\x01"),
    ("i8 min", -128, np.int8, b"\x80"),
    ("i8 max", 127, np.int8, b"\x7f"),
    ("i16 0", 0, np.int16, b"\x00\x00"),
    ("i16 1", 1, np.int16, b"\x01\x00"),
    ("i16 min", -32768, np.int16, b"\x00\x80"),
    ("i16 max", 32767, np.int16, b"\xff\x7f"),
    ("i32 0", 0, np.int32, b"\x00\x00\x00\x00"),
    ("i32 1", 1, np.int32, b"\x01\x00\x00\x00"),
    ("i32 min", -2147483648, np.int32, b"\x00\x00\x00\x80"),
    ("i32 max", 2147483647, np.int32, b"\xff\xff\xff\x7f"),
    ("i64 0", 0, np.int64, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
    ("i64 1", 1, np.int64, b"\x01\x00\x00\x00\x00\x00\x00\x00"),
    ("i64 min", -9223372036854775808, np.int64, b"\x00\x00\x00\x00\x00\x00\x00\x80"),
    ("i64 max", 9223372036854775807, np.int64, b"\xff\xff\xff\xff\xff\xff\xff\x7f"),
    ("u8 0", 0, np.uint8, b"\x00"),
    ("u8 1", 1, np.uint8, b"\x01"),
    ("u8 min", 0, np.uint8, b"\x00"),
    ("u8 max", 255, np.uint8, b"\xff"),
    ("u16 0", 0, np.uint16, b"\x00\x00"),
    ("u16 1", 1, np.uint16, b"\x01\x00"),
    ("u16 min", 0, np.uint16, b"\x00\x00"),
    ("u16 max", 65535, np.uint16, b"\xff\xff"),
    ("u32 0", 0, np.uint32, b"\x00\x00\x00\x00"),
    ("u32 1", 1, np.uint32, b"\x01\x00\x00\x00"),
    ("u32 min", 0, np.uint32, b"\x00\x00\x00\x00"),
    ("u32 max", 4294967295, np.uint32, b"\xff\xff\xff\xff"),
    ("u64 0", 0, np.uint64, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
    ("u64 1", 1, np.uint64, b"\x01\x00\x00\x00\x00\x00\x00\x00"),
    ("u64 min", 0, np.uint64, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
    ("u64 max", 18446744073709551615, np.uint64, b"\xff\xff\xff\xff\xff\xff\xff\xff"),
]


@pytest.mark.parametrize(
    "label,value,dtype,expected", INT_GOLDEN, ids=[row[0] for row in INT_GOLDEN]
)
def test_numpy_integer_matches_value_encoding_v1(label, value, dtype, expected):
    """The canonical cross-language integer grid: [0, 1, MIN, MAX] per dtype,
    little-endian (the Rust golden pins the same constants independently)."""
    got = np.array(value, dtype=dtype).tobytes()
    assert got == expected, (
        f"{label}: numpy {dtype.__name__} produced {got.hex()}, want {expected.hex()}"
    )


@pytest.mark.parametrize(
    "label,value,dtype,expected", INT_GOLDEN, ids=[row[0] for row in INT_GOLDEN]
)
def test_value_encoding_v1_integer_bytes_decode_to_the_golden_values(
    label, value, dtype, expected
):
    """Decode direction: the checked-in byte literals are the oracle, so a
    matched encode+decode drift (both sides moving together) still fails."""
    got = np.frombuffer(expected, dtype=dtype)[0]
    assert int(got) == value, (
        f"{label}: bytes {expected.hex()} decoded to {got}, want {value}"
    )


FLOAT_DECODE_GOLDEN: list[tuple[str, bytes, type, float]] = [
    ("f16 +0.0", b"\x00\x00", np.float16, 0.0),
    ("f16 -0.0", b"\x00\x80", np.float16, -0.0),
    ("f16 1.0", b"\x00\x3c", np.float16, 1.0),
    ("f16 +inf", b"\x00\x7c", np.float16, float("inf")),
    ("f16 -inf", b"\x00\xfc", np.float16, float("-inf")),
    ("f16 max finite", b"\xff\x7b", np.float16, 65504.0),
    ("f16 min subnormal", b"\x01\x00", np.float16, 5.9604644775390625e-8),
    ("f16 double-rounding", b"\x01\x3c", np.float16, 1.0009765625),
    ("f32 1.0", b"\x00\x00\x80\x3f", np.float32, 1.0),
    ("f32 -0.0", b"\x00\x00\x00\x80", np.float32, -0.0),
    ("f64 1.0", b"\x00\x00\x00\x00\x00\x00\xf0\x3f", np.float64, 1.0),
    ("f64 -0.0", b"\x00\x00\x00\x00\x00\x00\x00\x80", np.float64, -0.0),
]


@pytest.mark.parametrize(
    "label,data,dtype,expected",
    FLOAT_DECODE_GOLDEN,
    ids=[row[0] for row in FLOAT_DECODE_GOLDEN],
)
def test_value_encoding_v1_float_bytes_decode_to_the_golden_values(
    label, data, dtype, expected
):
    """Decode direction for the float grid (same oracle rule as the integer
    decode golden). ``0x3C01`` pins the exact f16 value ``1.0009765625``, the
    single-rounded result of ``DOUBLE_ROUNDING``. Equality alone cannot see a
    dropped sign on zero (``0.0 == -0.0``), hence the signbit assertion."""
    got = np.frombuffer(data, dtype=dtype)[0]
    assert float(got) == expected, (
        f"{label}: bytes {data.hex()} decoded to {got!r}, want {expected!r}"
    )
    assert np.signbit(got) == np.signbit(expected), (
        f"{label}: bytes {data.hex()} decoded with the wrong sign"
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
