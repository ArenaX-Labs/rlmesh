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
