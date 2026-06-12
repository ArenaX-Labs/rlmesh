from __future__ import annotations

import pytest


def test_tensor_validates_dtype_and_byte_length() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x00\x01", [2], "uint8")

    assert tensor.shape == [2]
    assert tensor.dtype == "uint8"
    assert tensor.ndim == 1
    assert tensor.size == 2
    assert tensor.nbytes == 2

    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        rlmesh.Tensor(b"\x00", [1], "complex64")

    with pytest.raises(ValueError, match="tensor byte length mismatch"):
        rlmesh.Tensor(b"\x00", [2], "uint8")


def test_tensor_strides_are_c_contiguous_byte_strides() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(bytes(range(24)), [2, 3], "float32")

    assert tensor.size == 6
    assert tensor.nbytes == 24
    assert tensor.strides == [12, 4]


def test_tensor_memoryview_is_read_only_and_tensor_owned() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x01\x02\x03", [3], "uint8")

    view = memoryview(tensor)
    assert view.readonly is True
    assert view.tobytes() == b"\x01\x02\x03"
    assert tensor.tobytes() == b"\x01\x02\x03"

    with pytest.raises(TypeError):
        view[0] = 9

    buffer_view = tensor.buffer
    assert isinstance(buffer_view, memoryview)
    assert buffer_view.readonly is True
    assert buffer_view.tobytes() == b"\x01\x02\x03"


def test_tensor_memoryview_is_n_dimensional_and_typed() -> None:
    import struct

    import rlmesh

    data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    tensor = rlmesh.Tensor(data, [2, 3], "float32")

    view = memoryview(tensor)
    assert view.format == "f"
    assert view.ndim == 2
    assert view.shape == (2, 3)
    assert view.strides == (12, 4)
    assert view.itemsize == 4
    assert view[1, 2] == 6.0
    assert view.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tensor_memoryview_formats_per_dtype() -> None:
    import rlmesh

    expected_formats = {
        "bool": "?",
        "int8": "b",
        "uint8": "B",
        "int16": "h",
        "uint16": "H",
        "int32": "i",
        "uint32": "I",
        "int64": "q",
        "uint64": "Q",
        "float16": "e",
        "float32": "f",
        "float64": "d",
    }
    sizes = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
        "float16": 2,
        "float32": 4,
        "float64": 8,
    }
    for dtype, fmt in expected_formats.items():
        tensor = rlmesh.Tensor(bytes(sizes[dtype]), [1], dtype)
        view = memoryview(tensor)
        assert view.format == fmt, dtype
        assert view.itemsize == sizes[dtype], dtype


def test_tensor_memoryview_int64_roundtrips_values() -> None:
    import struct

    import rlmesh

    data = struct.pack("<2q", -(2**62), 2**62)
    tensor = rlmesh.Tensor(data, [2], "int64")

    view = memoryview(tensor)
    assert view.tolist() == [-(2**62), 2**62]


def test_tensor_bfloat16_buffer_is_rejected() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x00\x3f", [1], "bfloat16")

    with pytest.raises(BufferError, match="__dlpack__"):
        memoryview(tensor)

    # tobytes() remains the supported byte-level escape hatch.
    assert tensor.tobytes() == b"\x00\x3f"


def test_tensor_scalar_memoryview() -> None:
    import struct

    import rlmesh

    tensor = rlmesh.Tensor(struct.pack("<d", 2.5), [], "float64")

    view = memoryview(tensor)
    assert view.ndim == 0
    assert view.shape == ()
    assert view.tolist() == 2.5


def test_tensor_simple_byte_consumers_still_work() -> None:
    import struct

    import rlmesh

    data = struct.pack("<3i", 1, 2, 3)
    tensor = rlmesh.Tensor(data, [3], "int32")

    # bytes() issues a PyBUF_SIMPLE request (no shape/strides/format).
    assert bytes(memoryview(tensor)) == data


def test_tensor_reshape_and_copy() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(bytes(range(6)), [2, 3], "uint8")

    reshaped = tensor.reshape([3, 2])
    assert reshaped.shape == [3, 2]
    assert reshaped.tobytes() == tensor.tobytes()
    assert reshaped.is_contiguous() is True

    copied = tensor.copy()
    assert copied.shape == [2, 3]
    assert copied.tobytes() == tensor.tobytes()

    assert tensor.device == "cpu"

    with pytest.raises(ValueError, match="reshape"):
        tensor.reshape([4, 2])
