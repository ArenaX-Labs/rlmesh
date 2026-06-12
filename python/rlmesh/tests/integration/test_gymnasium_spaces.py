from __future__ import annotations

import numpy as np
import pytest


def test_unrestricted_text_converts_to_gymnasium_printable_charset() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    text = spaces.Text(32)

    restored = spaces.to_gymnasium_space(text)

    assert isinstance(restored, gymnasium.spaces.Text)
    assert restored.contains("pick up the object!")


def test_default_gymnasium_text_imports_as_unrestricted_rlmesh_text() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Text(max_length=32)

    space = spaces.from_gymnasium_space(source)

    assert isinstance(space, spaces.Text)
    assert space.charset == ""
    assert space.contains("pick up the object!")


def test_gymnasium_box_roundtrip_preserves_stable_shape_dtype_and_bounds() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(2, 3),
        dtype=np.float32,
    )

    space = spaces.from_gymnasium_space(source)

    assert isinstance(space, spaces.Box)
    assert space.shape == [2, 3]
    assert space.dtype == "float32"
    assert space.low == -1.0
    assert space.high == 1.0

    restored = spaces.to_gymnasium_space(space)
    assert isinstance(restored, gymnasium.spaces.Box)
    assert restored.shape == (2, 3)
    assert restored.dtype == np.dtype("float32")
    np.testing.assert_array_equal(restored.low, source.low)
    np.testing.assert_array_equal(restored.high, source.high)


def test_gymnasium_discrete_roundtrip_preserves_start() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Discrete(5, start=2)

    space = spaces.from_gymnasium_space(source)

    assert isinstance(space, spaces.Discrete)
    assert space.n == 5
    assert space.start == 2
    assert space.dtype == "int64"

    restored = spaces.to_gymnasium_space(space)
    assert isinstance(restored, gymnasium.spaces.Discrete)
    assert restored.n == 5
    assert restored.start == 2


def test_gymnasium_text_roundtrip_preserves_finite_charset() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Text(max_length=12, charset="ab ")

    space = spaces.from_gymnasium_space(source)

    assert isinstance(space, spaces.Text)
    assert set(space.charset) == {"a", "b", " "}
    assert space.contains("a b")
    assert not space.contains("a c")

    restored = spaces.to_gymnasium_space(space)
    assert isinstance(restored, gymnasium.spaces.Text)
    assert restored.contains("a b")
    assert not restored.contains("a c")


def test_gymnasium_nested_structure_roundtrip() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Dict(
        {
            "obs": gymnasium.spaces.Box(
                low=0.0,
                high=255.0,
                shape=(2,),
                dtype=np.float32,
            ),
            "choice": gymnasium.spaces.Discrete(3),
            "pair": gymnasium.spaces.Tuple(
                (
                    gymnasium.spaces.Discrete(2),
                    gymnasium.spaces.Box(
                        low=-1.0,
                        high=1.0,
                        shape=(1,),
                        dtype=np.float64,
                    ),
                )
            ),
        }
    )

    space = spaces.from_gymnasium_space(source)

    assert isinstance(space, spaces.Dict)
    assert set(space.spaces) == {"choice", "obs", "pair"}
    assert isinstance(space.spaces["obs"], spaces.Box)
    assert isinstance(space.spaces["choice"], spaces.Discrete)
    assert isinstance(space.spaces["pair"], spaces.Tuple)

    restored = spaces.to_gymnasium_space(space)
    assert isinstance(restored, gymnasium.spaces.Dict)
    assert set(restored.spaces) == {"choice", "obs", "pair"}
    assert isinstance(restored.spaces["obs"], gymnasium.spaces.Box)
    assert isinstance(restored.spaces["choice"], gymnasium.spaces.Discrete)
    assert isinstance(restored.spaces["pair"], gymnasium.spaces.Tuple)


def test_gymnasium_int64_box_roundtrips_2_pow_63_minus_1_exactly() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    top = np.int64(2**63 - 1)
    source = gymnasium.spaces.Box(
        low=np.int64(0),
        high=top,
        shape=(2,),
        dtype=np.int64,
    )

    space = spaces.from_gymnasium_space(source)
    assert isinstance(space, spaces.Box)
    assert space.dtype == "int64"

    restored = spaces.to_gymnasium_space(space)
    assert isinstance(restored, gymnasium.spaces.Box)
    assert restored.dtype == np.dtype("int64")
    # The bound survives exactly; an f64 round-trip would round it up to 2**63.
    assert int(restored.high.max()) == 2**63 - 1
    np.testing.assert_array_equal(restored.high, source.high)
    np.testing.assert_array_equal(restored.low, source.low)


def test_gymnasium_int64_elementwise_box_roundtrips_exactly() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Box(
        low=np.array([0, 100], dtype=np.int64),
        high=np.array([10, 2**63 - 1], dtype=np.int64),
        dtype=np.int64,
    )

    space = spaces.from_gymnasium_space(source)
    restored = spaces.to_gymnasium_space(space)

    np.testing.assert_array_equal(restored.low, source.low)
    np.testing.assert_array_equal(restored.high, source.high)


def test_gymnasium_uint64_box_roundtrips_full_range_exactly() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Box(
        low=np.uint64(0),
        high=np.uint64(2**64 - 1),
        shape=(1,),
        dtype=np.uint64,
    )

    space = spaces.from_gymnasium_space(source)
    restored = spaces.to_gymnasium_space(space)

    assert restored.dtype == np.dtype("uint64")
    assert int(restored.high.max()) == 2**64 - 1
    np.testing.assert_array_equal(restored.high, source.high)


def test_gymnasium_shape_2_1_box_is_not_misclassified() -> None:
    # A (2,1) Box has numel == rank == 2; the retired Axiswise classification
    # used to mistake its elementwise bounds for per-axis bounds and produce a
    # spec that raised on reconstruction. It must now round-trip cleanly.
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    source = gymnasium.spaces.Box(
        low=np.array([[0.0], [1.0]]),
        high=np.array([[1.0], [2.0]]),
        shape=(2, 1),
        dtype=np.float32,
    )

    space = spaces.from_gymnasium_space(source)
    restored = spaces.to_gymnasium_space(space)

    assert isinstance(restored, gymnasium.spaces.Box)
    assert restored.shape == (2, 1)
    np.testing.assert_array_equal(restored.low, source.low)
    np.testing.assert_array_equal(restored.high, source.high)


def test_gymnasium_rank2_scalar_low_broadcast_matches_gymnasium() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    from rlmesh import spaces

    # gymnasium broadcasts the scalar low across the (2,3) shape; the rlmesh
    # round-trip must reproduce that exact per-element broadcast.
    source = gymnasium.spaces.Box(
        low=0.0,
        high=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        shape=(2, 3),
        dtype=np.float32,
    )

    space = spaces.from_gymnasium_space(source)
    restored = spaces.to_gymnasium_space(space)

    np.testing.assert_array_equal(restored.low, source.low)
    np.testing.assert_array_equal(restored.high, source.high)
