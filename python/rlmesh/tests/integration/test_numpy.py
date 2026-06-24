from __future__ import annotations

import numpy as np
import pytest


def test_numpy_tensor_roundtrip() -> None:
    from rlmesh import Tensor
    from rlmesh import numpy as rlmesh_numpy

    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = rlmesh_numpy.from_array(source)

    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 3]
    assert tensor.dtype == "float32"

    restored = rlmesh_numpy.asarray(tensor)
    np.testing.assert_array_equal(restored, source)


def test_numpy_scalar_from_array_returns_primitive() -> None:
    from rlmesh import numpy as rlmesh_numpy

    assert rlmesh_numpy.from_array(np.asarray(3, dtype=np.int64)) == 3


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (np.bool_, [[True, False], [False, True]]),
        (np.uint8, [[1, 2], [3, 4]]),
        (np.int32, [[1, -2], [3, -4]]),
        (np.int64, [[1, -2], [3, -4]]),
        (np.float16, [[1.5, -2.0], [3.25, -4.5]]),
        (np.float32, [[1.5, -2.0], [3.25, -4.5]]),
        (np.float64, [[1.5, -2.0], [3.25, -4.5]]),
    ],
)
def test_numpy_tensor_roundtrip_stable_dtypes(
    dtype: type[np.generic], values: list[list[object]]
) -> None:
    from rlmesh import Tensor
    from rlmesh import numpy as rlmesh_numpy

    source = np.asarray(values, dtype=dtype)
    tensor = rlmesh_numpy.from_array(source)

    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 2]
    assert tensor.dtype == str(source.dtype)

    restored = rlmesh_numpy.asarray(tensor)
    np.testing.assert_array_equal(restored, source)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (np.int8, [[-128, 0], [1, 127]]),
        (np.int16, [[-5, 0], [1, 999]]),
        (np.uint16, [[0, 1], [2, 65535]]),
        (np.uint32, [[0, 1], [2, 4_000_000_000]]),
        (np.uint64, [[0, 1], [2, 2**63]]),
    ],
)
def test_numpy_tensor_roundtrip_extended_dtypes(
    dtype: type[np.generic], values: list[list[object]]
) -> None:
    from rlmesh import Tensor
    from rlmesh import numpy as rlmesh_numpy

    source = np.asarray(values, dtype=dtype)
    tensor = rlmesh_numpy.from_array(source)

    assert isinstance(tensor, Tensor)
    assert tensor.dtype == str(source.dtype)

    restored = rlmesh_numpy.asarray(tensor)
    np.testing.assert_array_equal(restored, source)


def test_numpy_asarray_is_writable_copy() -> None:
    from rlmesh import Tensor
    from rlmesh import numpy as rlmesh_numpy

    tensor = Tensor(np.arange(4, dtype=np.float32).tobytes(), [4], "float32")
    array = rlmesh_numpy.asarray(tensor)

    # Matches Gymnasium: decoded observations are writable.
    assert array.flags.writeable is True
    array[0] = 1.0
    # Writing the decoded array must not corrupt the tensor buffer.
    np.testing.assert_array_equal(
        rlmesh_numpy.asarray(tensor),
        np.arange(4, dtype=np.float32),
    )
