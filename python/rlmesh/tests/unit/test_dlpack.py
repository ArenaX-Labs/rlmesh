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


def _np_version(np: object) -> tuple[int, int]:
    major, minor = str(np.__version__).split(".")[:2]  # type: ignore[attr-defined]
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


def test_numpy_from_dlpack_roundtrip() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    if _np_version(np) < (1, 23):
        pytest.skip("public np.from_dlpack requires numpy >= 1.23")

    data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    tensor = rlmesh.Tensor(data, [2, 3], "float32")

    array = np.from_dlpack(tensor)
    assert array.shape == (2, 3)
    assert array.dtype == np.float32
    assert array.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    # numpy >= 2.1 requests the versioned capsule and honors READ_ONLY.
    if _np_version(np) >= (2, 1):
        assert array.flags.writeable is False

    # The exported view shares memory with the tensor.
    base = np.from_dlpack(tensor)
    assert base.__array_interface__["data"][0] == array.__array_interface__["data"][0]


def test_numpy_from_dlpack_copy_is_writable() -> None:
    np = pytest.importorskip("numpy")
    import rlmesh

    if _np_version(np) < (2, 1):
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

    if _np_version(np) < (1, 23):
        pytest.skip("public np.from_dlpack requires numpy >= 1.23")

    int_tensor = rlmesh.Tensor(struct.pack("<2q", -1, 2**60), [2], "int64")
    assert np.from_dlpack(int_tensor).dtype == np.int64
    assert np.from_dlpack(int_tensor).tolist() == [-1, 2**60]

    if _np_version(np) >= (1, 25):  # numpy DLPack bool support
        bool_tensor = rlmesh.Tensor(b"\x01\x00", [2], "bool")
        assert np.from_dlpack(bool_tensor).dtype == np.bool_
        assert np.from_dlpack(bool_tensor).tolist() == [True, False]


def test_from_dlpack_roundtrip_legacy_and_versioned() -> None:
    import rlmesh

    data = struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    tensor = rlmesh.Tensor(data, [2, 3], "float32")

    # Object path: from_dlpack calls __dlpack__() with no kwargs -> legacy.
    imported = rlmesh.Tensor.from_dlpack(tensor)
    assert imported.shape == [2, 3]
    assert imported.dtype == "float32"
    assert imported.tobytes() == data

    # Raw capsule paths, both flavors.
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

    # A consumed capsule is no longer importable.
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

    # Strided source: every other column is a non-contiguous view.
    view = array[:, ::2]
    tensor = rlmesh.Tensor.from_dlpack(view)
    assert tensor.shape == [3, 2]
    assert tensor.is_contiguous() is True
    assert tensor.tobytes() == np.ascontiguousarray(view).tobytes()

    if _np_version(np) >= (1, 25):  # numpy DLPack bool support
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
    # The import copies and releases the producer immediately: no lingering
    # reference to the numpy array survives.
    assert sys.getrefcount(array) == refcount_before

    array[0] = 99.0
    assert tensor.tobytes() == struct.pack("<4d", 0.0, 1.0, 2.0, 3.0)


def test_from_dlpack_rejects_non_dlpack_objects() -> None:
    import rlmesh

    with pytest.raises(TypeError, match="__dlpack__"):
        rlmesh.Tensor.from_dlpack(object())

    with pytest.raises(TypeError, match="__dlpack__"):
        rlmesh.Tensor.from_dlpack(b"raw bytes")


def test_from_dlpack_rejects_non_cpu_devices() -> None:
    """Fabricate a kDLCUDA capsule with ctypes and check it is refused."""
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
    managed = DLManagedTensor()
    managed.dl_tensor.data = ctypes.cast(payload, ctypes.c_void_p)
    managed.dl_tensor.device = DLDevice(device_type=2, device_id=0)  # kDLCUDA
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = DLDataType(code=2, bits=32, lanes=1)
    managed.dl_tensor.shape = shape
    managed.dl_tensor.byte_offset = 0
    managed.deleter = None

    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = new_capsule(
        ctypes.cast(ctypes.byref(managed), ctypes.c_void_p), b"dltensor", None
    )

    with pytest.raises(ValueError, match="only CPU"):
        rlmesh.Tensor.from_dlpack(capsule)


def _rss_bytes() -> int:
    """Resident set size of this process, in bytes.

    Linux exposes current RSS via ``/proc/self/statm``. Elsewhere (macOS/BSD)
    fall back to peak RSS from ``getrusage`` -- for this leak test that is an
    equally valid signal: a holder or storage leak keeps the high-water mark
    climbing across the stress loop, while a leak-free run holds it flat.
    """
    import sys

    if sys.platform.startswith("linux"):
        import os

        page_size = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * page_size

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kibibytes on Linux/BSD.
    return peak if sys.platform == "darwin" else peak * 1024


def test_dlpack_capsule_lifecycle_does_not_leak() -> None:
    """Stress every capsule lifecycle path and assert RSS stays flat.

    Each export allocates a holder plus a 4 KiB storage refcount; a leak in
    any of the paths below would grow RSS by hundreds of MiB over the run,
    far beyond the assertion threshold.
    """
    import gc

    import rlmesh

    # High rank on purpose: the export holder owns shape/strides vectors, so
    # a leaked holder costs ~1 KiB here and trips the threshold even when
    # the (shared) storage itself is not leaked.
    shape = [1] * 60 + [1024]
    tensor = rlmesh.Tensor(bytes(4096), shape, "float32")

    def exercise(iterations: int) -> None:
        for _ in range(iterations):
            # Export and drop unconsumed, both capsule flavors: the capsule
            # destructor must free the holder.
            legacy = tensor.__dlpack__()
            versioned = tensor.__dlpack__(max_version=(1, 0))
            del legacy, versioned
            # Export consumed by our own importer: the consumer must free
            # the holder via the producer deleter after renaming.
            imported = rlmesh.Tensor.from_dlpack(tensor.__dlpack__())
            # A copying export allocates fresh storage that must also die.
            copied = tensor.__dlpack__(copy=True, max_version=(1, 0))
            del imported, copied

    exercise(1_000)  # warmup: allocator pools, import caches
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
