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


def _version(module: object) -> tuple[int, int]:
    major, minor = str(module.__version__).split(".")[:2]  # type: ignore[attr-defined]
    return int(major), int(minor)


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

    del capsule


def test_dlpack_versioned_capsule_when_requested() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(struct.pack("<2f", 1.0, 2.0), [2], "float32")

    capsule = tensor.__dlpack__(max_version=(1, 0))
    assert _capsule_name_is(capsule, b"dltensor_versioned")

    old = tensor.__dlpack__(max_version=(0, 8))
    assert _capsule_name_is(old, b"dltensor")


def test_dlpack_rejects_stream_and_foreign_devices() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x00", [1], "uint8")

    with pytest.raises(BufferError, match="stream"):
        tensor.__dlpack__(stream=1)

    with pytest.raises(BufferError, match="CPU"):
        tensor.__dlpack__(dl_device=(2, 0))

    capsule = tensor.__dlpack__(dl_device=(1, 0))
    assert _capsule_name_is(capsule, b"dltensor")


def test_dlpack_copy_exports_fresh_buffer() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(struct.pack("<2f", 1.0, 2.0), [2], "float32")

    capsule = tensor.__dlpack__(copy=True, max_version=(1, 0))
    assert _capsule_name_is(capsule, b"dltensor_versioned")

    class VersionedHeader(ctypes.Structure):
        _fields_: ClassVar = [
            ("major", ctypes.c_uint32),
            ("minor", ctypes.c_uint32),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
            ("flags", ctypes.c_uint64),
        ]

    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.restype = ctypes.c_void_p
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    for copy, flags in [(True, 2), (False, 1)]:
        capsule = tensor.__dlpack__(copy=copy, max_version=(1, 0))
        header = ctypes.cast(
            get_pointer(capsule, b"dltensor_versioned"), ctypes.POINTER(VersionedHeader)
        ).contents
        assert (header.major, header.minor, header.flags) == (1, 0, flags)


def test_numpy_from_dlpack_roundtrip() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    if _version(np) < (1, 23):
        pytest.skip("public np.from_dlpack requires numpy >= 1.23")

    data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    tensor = rlmesh.Tensor(data, [2, 3], "float32")

    array = np.from_dlpack(tensor)
    assert array.shape == (2, 3)
    assert array.dtype == np.float32
    assert array.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    if _version(np) >= (2, 1):
        assert array.flags.writeable is False

    base = np.from_dlpack(tensor)
    assert base.__array_interface__["data"][0] == array.__array_interface__["data"][0]


def test_numpy_from_dlpack_copy_is_writable() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    if _version(np) < (2, 1):
        pytest.skip("np.from_dlpack(copy=...) requires numpy >= 2.1")

    tensor = rlmesh.Tensor(struct.pack("<2q", 7, 8), [2], "int64")

    array = np.from_dlpack(tensor, copy=True)
    assert array.flags.writeable is True
    array[0] = 42
    assert array.tolist() == [42, 8]
    assert tensor.tobytes() == struct.pack("<2q", 7, 8)


def test_numpy_sees_int64_and_bool_dtypes() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    if _version(np) < (1, 23):
        pytest.skip("public np.from_dlpack requires numpy >= 1.23")

    int_tensor = rlmesh.Tensor(struct.pack("<2q", -1, 2**60), [2], "int64")
    assert np.from_dlpack(int_tensor).dtype == np.int64
    assert np.from_dlpack(int_tensor).tolist() == [-1, 2**60]

    if _version(np) >= (1, 25):
        bool_tensor = rlmesh.Tensor(b"\x01\x00", [2], "bool")
        assert np.from_dlpack(bool_tensor).dtype == np.bool_
        assert np.from_dlpack(bool_tensor).tolist() == [True, False]


def test_from_dlpack_roundtrip_legacy_and_versioned() -> None:
    import rlmesh

    data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    tensor = rlmesh.Tensor(data, [2, 3], "float32")

    imported = rlmesh.Tensor.from_dlpack(tensor)
    assert imported.shape == [2, 3]
    assert imported.dtype == "float32"
    assert imported.tobytes() == data

    legacy = rlmesh.Tensor.from_dlpack(tensor.__dlpack__())
    assert legacy.tobytes() == data
    versioned = rlmesh.Tensor.from_dlpack(tensor.__dlpack__(max_version=(1, 0)))
    assert versioned.tobytes() == data


def test_from_dlpack_consumes_capsule_once() -> None:
    import rlmesh

    tensor = rlmesh.Tensor(b"\x01\x02", [2], "uint8")
    capsule = tensor.__dlpack__()

    first = rlmesh.Tensor.from_dlpack(capsule)
    assert first.tobytes() == b"\x01\x02"
    assert _capsule_name_is(capsule, b"used_dltensor")

    with pytest.raises(TypeError):
        rlmesh.Tensor.from_dlpack(capsule)

    versioned = tensor.__dlpack__(max_version=(1, 0))
    rlmesh.Tensor.from_dlpack(versioned)
    assert _capsule_name_is(versioned, b"used_dltensor_versioned")


def test_from_dlpack_imports_numpy_arrays() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    array = np.arange(12, dtype=np.int32).reshape(3, 4)
    tensor = rlmesh.Tensor.from_dlpack(array)
    assert tensor.shape == [3, 4]
    assert tensor.dtype == "int32"
    assert tensor.tobytes() == array.tobytes()

    view = array[:, ::2]
    tensor = rlmesh.Tensor.from_dlpack(view)
    assert tensor.shape == [3, 2]
    assert tensor.is_contiguous() is True
    assert tensor.tobytes() == np.ascontiguousarray(view).tobytes()

    if _version(np) >= (1, 25):
        bools = np.array([True, False, True])
        tensor = rlmesh.Tensor.from_dlpack(bools)
        assert tensor.dtype == "bool"
        assert tensor.tobytes() == b"\x01\x00\x01"


def test_from_dlpack_copies_out_of_the_source() -> None:
    np = pytest.importorskip("numpy")
    import sys

    import rlmesh

    array = np.arange(4, dtype=np.float64)
    refcount_before = sys.getrefcount(array)
    tensor = rlmesh.Tensor.from_dlpack(array)
    assert sys.getrefcount(array) == refcount_before

    array[0] = 99.0
    assert tensor.tobytes() == struct.pack("<4d", 0.0, 1.0, 2.0, 3.0)


def test_from_dlpack_rejects_non_dlpack_objects() -> None:
    import rlmesh

    with pytest.raises(TypeError, match="__dlpack__"):
        rlmesh.Tensor.from_dlpack(object())

    with pytest.raises(TypeError, match="__dlpack__"):
        rlmesh.Tensor.from_dlpack(b"raw bytes")


@pytest.mark.parametrize("versioned", [False, True])
def test_from_dlpack_rejects_non_cpu_devices(versioned: bool) -> None:
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

    class DLManagedTensor(ctypes.Structure):
        _fields_: ClassVar = [
            ("dl_tensor", DLTensor),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
        ]

    shape = (ctypes.c_int64 * 1)(1)
    payload = (ctypes.c_uint8 * 4)(0, 0, 0, 0)

    class DLPackVersion(ctypes.Structure):
        _fields_: ClassVar = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]

    class DLManagedTensorVersioned(ctypes.Structure):
        _fields_: ClassVar = [
            ("version", DLPackVersion),
            ("manager_ctx", ctypes.c_void_p),
            ("deleter", ctypes.c_void_p),
            ("flags", ctypes.c_uint64),
            ("dl_tensor", DLTensor),
        ]

    if versioned:
        managed = DLManagedTensorVersioned()
        managed.version = DLPackVersion(1, 0)
    else:
        managed = DLManagedTensor()
    name = b"dltensor_versioned" if versioned else b"dltensor"
    used = b"used_dltensor_versioned" if versioned else b"used_dltensor"
    managed.dl_tensor.data = ctypes.cast(payload, ctypes.c_void_p)
    managed.dl_tensor.device = DLDevice(device_type=2, device_id=0)
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = DLDataType(code=2, bits=32, lanes=1)
    managed.dl_tensor.shape = shape
    managed.dl_tensor.byte_offset = 0
    calls = []

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def deleter(_pointer: int) -> None:
        calls.append("deleted")

    managed.deleter = ctypes.cast(deleter, ctypes.c_void_p)

    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = new_capsule(
        ctypes.cast(ctypes.byref(managed), ctypes.c_void_p), name, None
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="only CPU"):
            rlmesh.Tensor.from_dlpack(capsule)
        assert calls == []
        assert _capsule_name_is(capsule, name)

    managed.dl_tensor.device.device_type = 1
    imported = rlmesh.Tensor.from_dlpack(capsule)
    assert imported.tobytes() == bytes(4)
    assert calls == ["deleted"]
    assert _capsule_name_is(capsule, used)
    with pytest.raises(TypeError):
        rlmesh.Tensor.from_dlpack(capsule)
    del capsule, imported
    assert calls == ["deleted"]


def _rss_bytes() -> int:
    import sys

    if sys.platform.startswith("linux"):
        import os

        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * page_size

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def test_dlpack_capsule_lifecycle_does_not_leak() -> None:
    import gc

    import rlmesh

    shape = [1] * 60 + [1024]
    tensor = rlmesh.Tensor(bytes(4096), shape, "float32")

    def exercise(iterations: int) -> None:
        for _ in range(iterations):
            legacy = tensor.__dlpack__()
            versioned = tensor.__dlpack__(max_version=(1, 0))
            del legacy, versioned
            imported = rlmesh.Tensor.from_dlpack(tensor.__dlpack__())
            copied = tensor.__dlpack__(copy=True, max_version=(1, 0))
            del imported, copied

    exercise(1_000)
    gc.collect()
    baseline = _rss_bytes()

    exercise(20_000)
    gc.collect()
    growth = _rss_bytes() - baseline

    assert growth < 32 * 1024 * 1024, (
        f"RSS grew {growth / 1024 / 1024:.1f} MiB across 20k capsule "
        "lifecycles; a holder or storage leak is likely"
    )


def test_dlpack_capsule_data_pointer_and_strides() -> None:
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
    assert [managed.strides[i] for i in range(2)] == [3, 1]
    assert managed.device.device_type == 1


@pytest.mark.parametrize(
    "dtype",
    ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"],
)
def test_numpy_dlpack_dtype_and_source_lifetimes(dtype: str) -> None:
    np = pytest.importorskip("numpy")
    import gc

    import rlmesh

    if _version(np) < (1, 25):
        pytest.skip("all scalar DLPack dtypes require numpy >= 1.25")
    source = np.array([0, 1, 1, 0], dtype=dtype).reshape(2, 2)
    expected = source.tobytes()
    tensor = rlmesh.Tensor.from_dlpack(source)
    del source
    gc.collect()
    assert tensor.dtype == dtype
    assert tensor.tobytes() == expected

    array = np.from_dlpack(tensor)
    del tensor
    gc.collect()
    assert array.dtype == np.dtype(dtype)
    assert array.tobytes() == expected


@pytest.mark.parametrize(
    "layout", ["transpose", "reverse", "broadcast", "empty", "scalar"]
)
def test_numpy_dlpack_strides(layout: str) -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    source = np.arange(12, dtype=np.int32).reshape(3, 4)
    if layout == "transpose":
        source = source.T
    elif layout == "reverse":
        source = source[::-1, ::-2]
    elif layout == "broadcast":
        source = np.lib.stride_tricks.as_strided(source[0, :2], (3, 2), (0, 4))
    elif layout == "empty":
        source = source[:0]
    else:
        source = np.array(42, dtype=np.int32)
    tensor = rlmesh.Tensor.from_dlpack(source)
    assert tensor.shape == list(source.shape)
    assert tensor.is_contiguous()
    assert tensor.tobytes() == source.tobytes()
    if source.size:
        expected_strides = (
            list(np.ascontiguousarray(source).strides) if source.ndim else []
        )
        assert tensor.strides == expected_strides


@pytest.mark.parametrize("dtype", ["bool", "int8", "int16", "int32", "int64", "uint8"])
def test_torch_dlpack_roundtrip_and_lifetimes(dtype: str) -> None:
    torch = pytest.importorskip("torch")
    import gc

    import rlmesh

    if dtype == "bool" and _version(torch) < (2, 0):
        pytest.skip("torch DLPack bool support requires torch >= 2.0")

    source = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=getattr(torch, dtype))
    view = source.T[:, ::2]
    tensor = rlmesh.Tensor.from_dlpack(view)
    expected = view.tolist()
    del source, view
    gc.collect()
    assert tensor.dtype == dtype
    for copy in (False, True):
        exported = torch.utils.dlpack.from_dlpack(tensor.__dlpack__(copy=copy))
        assert exported.dtype == getattr(torch, dtype)
        assert exported.tolist() == expected
    del tensor
    gc.collect()
    assert exported.tolist() == expected
