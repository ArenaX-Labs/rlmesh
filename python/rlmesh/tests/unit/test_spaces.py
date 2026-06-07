from __future__ import annotations


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

    assert spec._details()["n"] == 3
    assert spec._to_dict()["kind"] == "discrete"
    assert repr(space) == "Discrete(kind='discrete', shape=[], dtype='int64')"


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


def test_text_default_is_unrestricted_and_explicit_charset_is_restrictive() -> None:
    from rlmesh import spaces

    text = spaces.Text(32)

    assert text.charset == ""
    assert text.contains("pick up the object!")
    assert text.contains(text.sample())

    finite = spaces.Text(32, charset="abc")
    assert finite.contains("abc")
    assert not finite.contains("a b")
