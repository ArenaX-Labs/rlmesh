from __future__ import annotations

import ctypes
import struct
from typing import ClassVar

import pytest

_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _capsule_name_is(capsule: object, name: bytes) -> bool:
    return _PyCapsule_IsValid(capsule, name) == 1


def test_dlpack_device_is_cpu() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x00", [1], "uint8")
    assert tensor.__dlpack_device__() == (1, 0)


def test_dlpack_legacy_capsule_by_default() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(struct.pack("<2f", 1.0, 2.0), [2], "float32")

    capsule = tensor.__dlpack__()
    assert _capsule_name_is(capsule, b"dltensor")
    assert not _capsule_name_is(capsule, b"dltensor_versioned")

    # Dropping an unconsumed capsule must free the export without crashing.
    del capsule


def test_dlpack_versioned_capsule_when_requested() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(struct.pack("<2f", 1.0, 2.0), [2], "float32")

    capsule = tensor.__dlpack__(max_version=(1, 0))
    assert _capsule_name_is(capsule, b"dltensor_versioned")

    # Pre-1.0 consumers still get the legacy capsule.
    old = tensor.__dlpack__(max_version=(0, 8))
    assert _capsule_name_is(old, b"dltensor")


def test_dlpack_rejects_stream_and_foreign_devices() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x00", [1], "uint8")

    with pytest.raises(BufferError, match="stream"):
        tensor.__dlpack__(stream=1)

    with pytest.raises(BufferError, match="CPU"):
        tensor.__dlpack__(dl_device=(2, 0))  # kDLCUDA

    # The tensor's own device is accepted.
    capsule = tensor.__dlpack__(dl_device=(1, 0))
    assert _capsule_name_is(capsule, b"dltensor")


def test_dlpack_copy_exports_fresh_buffer() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(struct.pack("<2f", 1.0, 2.0), [2], "float32")

    capsule = tensor.__dlpack__(copy=True, max_version=(1, 0))
    assert _capsule_name_is(capsule, b"dltensor_versioned")


def test_dlpack_bfloat16_exports() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x00\x3f", [1], "bfloat16")

    capsule = tensor.__dlpack__()
    assert _capsule_name_is(capsule, b"dltensor")


def test_numpy_from_dlpack_roundtrip() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    tensor = rlmesh.Tensor(data, [2, 3], "float32")

    array = np.from_dlpack(tensor)
    assert array.shape == (2, 3)
    assert array.dtype == np.float32
    assert array.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    # numpy >= 2.1 requests the versioned capsule and honors READ_ONLY.
    np_major, np_minor = (int(part) for part in np.__version__.split(".")[:2])
    if (np_major, np_minor) >= (2, 1):
        assert array.flags.writeable is False

    # The exported view shares memory with the tensor.
    base = np.from_dlpack(tensor)
    assert base.__array_interface__["data"][0] == array.__array_interface__["data"][0]


def test_numpy_from_dlpack_copy_is_writable() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    np_major, np_minor = (int(part) for part in np.__version__.split(".")[:2])
    if (np_major, np_minor) < (2, 1):
        pytest.skip("np.from_dlpack(copy=...) requires numpy >= 2.1")

    tensor = rlmesh.Tensor(struct.pack("<2q", 7, 8), [2], "int64")

    array = np.from_dlpack(tensor, copy=True)
    assert array.flags.writeable is True
    array[0] = 42
    assert array.tolist() == [42, 8]
    # The original tensor bytes are untouched.
    assert tensor.tobytes() == struct.pack("<2q", 7, 8)


def test_numpy_sees_int64_and_bool_dtypes() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    int_tensor = rlmesh.Tensor(struct.pack("<2q", -1, 2**60), [2], "int64")
    assert np.from_dlpack(int_tensor).dtype == np.int64
    assert np.from_dlpack(int_tensor).tolist() == [-1, 2**60]

    bool_tensor = rlmesh.Tensor(b"\x01\x00", [2], "bool")
    assert np.from_dlpack(bool_tensor).dtype == np.bool_
    assert np.from_dlpack(bool_tensor).tolist() == [True, False]


def test_dlpack_capsule_data_pointer_and_strides() -> None:
    """Walk the legacy DLManagedTensor struct via ctypes and check fields."""
    import rlmesh

    class DLDevice(ctypes.Structure):
        _fields_: ClassVar = [
            ("device_type", ctypes.c_int32),
            ("device_id", ctypes.c_int32),
        ]

    class DLDataType(ctypes.Structure):
        _fields_: ClassVar = [
            ("code", ctypes.c_uint8),
            ("bits", ctypes.c_uint8),
            ("lanes", ctypes.c_uint16),
        ]

    class DLTensor(ctypes.Structure):
        _fields_: ClassVar = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int32),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    tensor = rlmesh.Tensor(bytes(range(24)), [2, 3], "float32")
    capsule = tensor.__dlpack__()

    managed = ctypes.cast(
        get_pointer(capsule, b"dltensor"), ctypes.POINTER(DLTensor)
    ).contents
    assert managed.ndim == 2
    assert managed.byte_offset == 0
    assert (managed.dtype.code, managed.dtype.bits, managed.dtype.lanes) == (2, 32, 1)
    assert [managed.shape[i] for i in range(2)] == [2, 3]
    # Element-unit strides, C-order.
    assert [managed.strides[i] for i in range(2)] == [3, 1]
    assert managed.device.device_type == 1
