from __future__ import annotations

import pytest


def test_asarray_returns_writable_array() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh._rlmesh import Tensor
    from rlmesh.numpy import asarray, from_array

    tensor = from_array(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert isinstance(tensor, Tensor)
    array = asarray(tensor)

    assert array.flags.writeable is True
    # The Gymnasium preprocessing idiom must not raise on a read-only array.
    array /= 255.0
    assert array.shape == (2, 3)


def test_asarray_copy_is_independent_of_tensor() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh._rlmesh import Tensor
    from rlmesh.numpy import asarray, from_array

    source = np.arange(4, dtype=np.int64)
    tensor = from_array(source)
    assert isinstance(tensor, Tensor)

    array = asarray(tensor)
    array[0] = 999

    # Mutating the decoded array must not corrupt the tensor buffer.
    np.testing.assert_array_equal(asarray(tensor), source)
