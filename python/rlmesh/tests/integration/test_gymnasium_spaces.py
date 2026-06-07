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
