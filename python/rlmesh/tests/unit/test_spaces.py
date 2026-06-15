from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    import numpy as np

    NumpyArray = np.ndarray[Any, Any]


def test_space_spec_debug_shape_is_private() -> None:
    from rlmesh import spaces

    space = spaces.Discrete(3, start=1)
    spec = space.spec

    assert spec.kind == "discrete"
    assert spec.shape == []
    assert spec.dtype == "int64"
    assert space.n == 3
    assert space.start == 1

    assert not hasattr(spec, "details")
    assert not hasattr(spec, "to_dict")
    assert not hasattr(space, "details")
    assert not hasattr(space, "to_dict")

    details = cast(dict[str, object], spec._details())
    assert details["n"] == 3
    assert spec._to_dict()["kind"] == "discrete"
    assert repr(space) == "Discrete(kind='discrete', shape=[], dtype='int64')"


def test_native_space_seed_accepts_no_argument() -> None:
    from rlmesh import spaces

    # The native _rlmesh.Space is public via spec.to_space(); seed() must accept
    # a missing argument for gym parity (PyO3 0.23+ requires an explicit
    # signature default for trailing Option params).
    native = spaces.Discrete(3).spec.to_space()

    first = native.seed()
    assert isinstance(first, int)
    explicit = native.seed(123)
    assert explicit == 123


def test_space_family_uses_typed_wrapper_properties() -> None:
    from rlmesh import spaces

    box = spaces.Box(-1.0, 1.0, shape=[2], dtype="float32")
    multi_binary = spaces.MultiBinary([2, 3])
    multi_discrete = spaces.MultiDiscrete([2, 3], dtype="int64")
    text = spaces.Text(8, min_length=1)
    tuple_space = spaces.Tuple([box, spaces.Discrete(2)])

    assert box.kind == "box"
    assert box.shape == [2]
    assert box.dtype == "float32"
    assert box.low == -1.0
    assert box.high == 1.0

    assert multi_binary.kind == "multi_binary"
    assert multi_binary.shape == [2, 3]
    assert multi_binary.dims == [2, 3]

    assert multi_discrete.kind == "multi_discrete"
    assert multi_discrete.nvec == [2, 3]

    assert text.kind == "text"
    assert text.min_length == 1
    assert text.max_length == 8
    assert isinstance(text.charset, str)

    assert tuple_space.kind == "tuple"
    assert len(tuple_space.spaces) == 2

    source = spaces.Dict(
        {
            "obs": box,
            "action": spaces.Discrete(2),
        }
    )
    roundtripped = spaces.space_from_spec(source.spec)

    assert isinstance(roundtripped, spaces.Dict)
    assert set(roundtripped.spaces) == {"action", "obs"}
    assert isinstance(roundtripped.spaces["obs"], spaces.Box)
    assert roundtripped.spaces["obs"].shape == [2]
    assert roundtripped == source


def test_native_tensor_like_samples_use_canonical_tensor_values() -> None:
    from rlmesh import Tensor, spaces

    space = spaces.Box(-1.0, 1.0, shape=[2], dtype="float32")

    sample = space.sample()

    assert isinstance(sample, Tensor)
    assert sample.shape == [2]
    assert sample.dtype == "float32"
    assert space.contains(sample)


def test_numpy_space_from_spec_samples_and_contains_numpy_values() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh import numpy as rlmesh_numpy
    from rlmesh import spaces

    space = rlmesh_numpy.space_from_spec(
        spaces.Box(-1.0, 1.0, shape=[2], dtype="float32").spec
    )

    sample = space.sample()

    assert isinstance(sample, np.ndarray)
    sample_array = cast("NumpyArray", sample)
    assert sample_array.shape == (2,)
    assert sample_array.dtype == np.dtype("float32")
    assert space.contains(sample_array)
    assert space.contains(np.asarray([0.25, -0.25], dtype=np.float32))


def test_text_default_is_unrestricted_and_explicit_charset_is_restrictive() -> None:
    from rlmesh import spaces

    text = spaces.Text(32)

    assert text.charset == ""
    assert text.contains("pick up the object!")
    assert text.contains(text.sample())

    finite = spaces.Text(32, charset="abc")
    assert finite.contains("abc")
    assert not finite.contains("a b")


@pytest.mark.parametrize("dtype", ["int8", "uint64", "bfloat16"])
def test_native_box_samples_support_extended_dtypes(dtype: str) -> None:
    from rlmesh import Tensor, spaces

    space = spaces.Box(0.0, 10.0, shape=[2], dtype=dtype)

    sample = space.sample()

    assert isinstance(sample, Tensor)
    assert sample.dtype == dtype
    assert sample.shape == [2]
    assert space.contains(sample)
