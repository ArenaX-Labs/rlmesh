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
    assert repr(space) == "Discrete(3, start=1)"


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


def test_native_tensor_supports_numpy_array_protocol_zero_copy() -> None:
    np = pytest.importorskip("numpy")
    from rlmesh import spaces

    sample = spaces.Box(-1.0, 1.0, shape=[2, 3], dtype="float32").sample()

    array = np.asarray(sample)
    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 3)
    assert array.dtype == np.dtype("float32")
    # Shares the tensor's buffer rather than copying through tobytes().
    assert array.base is not None
    # dtype/copy follow the array protocol.
    assert np.asarray(sample, dtype=np.float64).dtype == np.dtype("float64")
    assert np.array(sample, copy=True).flags.writeable


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


@pytest.mark.parametrize("dtype", ["int8", "uint64"])
def test_native_box_samples_support_extended_dtypes(dtype: str) -> None:
    from rlmesh import Tensor, spaces

    space = spaces.Box(0.0, 10.0, shape=[2], dtype=dtype)

    sample = space.sample()

    assert isinstance(sample, Tensor)
    assert sample.dtype == dtype
    assert sample.shape == [2]
    assert space.contains(sample)


def test_dtype_name_maps_builtin_float_and_int() -> None:
    from rlmesh.spaces._internals import dtype_name

    # numpy-fluent dtype=float / int read as float64 / int64; the bare __name__
    # ('float'/'int') would be rejected by the Rust DType validator. bool already
    # resolves via __name__.
    assert dtype_name(float) == "float64"
    assert dtype_name(int) == "int64"
    assert dtype_name(bool) == "bool"


def test_box_accepts_builtin_float_dtype() -> None:
    from rlmesh import spaces

    assert spaces.Box(-1.0, 1.0, shape=[3], dtype=float).spec.dtype == "float64"


def test_space_underscore_getattr_is_plain_miss() -> None:
    from rlmesh import spaces

    space = spaces.Discrete(3)

    # Dunder/underscore probes (copy/pickle/inspect) get a clean AttributeError
    # without the Gymnasium-conversion hint; public misses keep the hint.
    with pytest.raises(AttributeError) as private_miss:
        _ = space._not_a_real_attr  # pyright: ignore[reportAttributeAccessIssue]
    assert "Gymnasium" not in str(private_miss.value)

    with pytest.raises(AttributeError, match="to_gymnasium_space"):
        _ = space.nvec  # pyright: ignore[reportAttributeAccessIssue]


def test_require_float_rejects_nan_keeps_inf() -> None:
    from rlmesh.spaces._internals import require_float

    # 'inf'/'-inf' are valid Box bounds; 'nan' makes every contains()/clamp compare
    # False, so it is rejected at the boundary.
    assert require_float("inf", "high") == float("inf")
    assert require_float("-inf", "low") == float("-inf")
    with pytest.raises(ValueError, match="NaN"):
        require_float("nan", "low")


def test_box_integer_bounds_are_exact_to_the_full_64_bit_range() -> None:
    from rlmesh import spaces

    # An exact Python int must not round-trip through a double: float() would
    # round anything above 2**53 and saturate a uint64 bound at i64::MAX, which
    # is the full-range exactness docs/compatibility.md promises.
    assert spaces.Box(0, 2**64 - 1, shape=[1], dtype="uint64").high == 2**64 - 1
    int64 = spaces.Box(-(2**63), 2**63 - 1, shape=[1], dtype="int64")
    assert int64.low == -(2**63)
    assert int64.high == 2**63 - 1
    assert spaces.Box(0, 2**53 + 1, shape=[1], dtype="int64").high == 2**53 + 1


def test_box_float_bounds_are_unchanged() -> None:
    from rlmesh import spaces

    # Floats keep the double path, an integral float bound on an integer dtype
    # reads as that integer, and the fully open range stays unbounded.
    box = spaces.Box(-1.5, 1.5, shape=[2], dtype="float32")
    assert box.bounds_kind == "uniform"
    assert box.low == -1.5
    assert box.high == 1.5
    assert spaces.Box(0.0, 10.0, shape=[1], dtype="int64").high == 10
    unbounded = spaces.Box(float("-inf"), float("inf"), shape=[1], dtype="float32")
    assert unbounded.bounds_kind == "unbounded"
    assert spaces.Box("-inf", 1.0, shape=[1], dtype="float32").low == float("-inf")


def test_box_rejects_nan_bounds() -> None:
    from rlmesh import spaces

    for low, high in ((float("nan"), 1.0), (0.0, float("nan"))):
        with pytest.raises(ValueError, match="NaN"):
            spaces.Box(low, high, shape=[1], dtype="float32")


def test_space_seed_always_returns_the_seed() -> None:
    from rlmesh import spaces

    # SPC-P1: seed() returns the seed it applied, drawn when None is given.
    box = spaces.Box(-1.0, 1.0, shape=[2])
    drawn = box.seed()
    assert isinstance(drawn, int)
    assert box.seed(2**64 - 1) == 2**64 - 1


@pytest.mark.parametrize(
    ("cls", "spec_kind"),
    [
        ("Box", "discrete"),
        ("Discrete", "box"),
        ("Dict", "box"),
        ("MultiBinary", "box"),
        ("MultiDiscrete", "box"),
        ("Text", "box"),
        ("Tuple", "box"),
    ],
)
def test_from_spec_refuses_a_spec_of_another_kind(cls: str, spec_kind: str) -> None:
    from rlmesh import spaces

    # SPC-P2 / SPC-P10: every wrapper refuses another kind with ValueError
    # (Box used to wrap any kind; Discrete and Text failed with KeyError).
    spec = (
        spaces.Discrete(3).spec
        if spec_kind == "discrete"
        else spaces.Box(0.0, 1.0, shape=[1]).spec
    )
    with pytest.raises(ValueError, match="SpaceSpec"):
        getattr(spaces, cls)(spec)


def test_typed_uniform_box_bounds_read_as_scalars() -> None:
    from rlmesh import spaces

    # SPC-P5: a typed uniform Box reads low/high as one number, as uniform does.
    box = spaces.Box(0, 255, shape=[2, 2], dtype="uint8")
    assert box.bounds_kind == "typed_uniform"
    assert box.low == 0
    assert box.high == 255
    # The describe-envelope dict form keeps its one-element list.
    details = cast(dict[str, object], box.spec._to_dict()["details"])
    assert details["low"] == [0]


@pytest.mark.parametrize("dtype", ["int32", "uint8", "bool"])
def test_box_refuses_fractional_bounds_on_integral_dtypes(dtype: str) -> None:
    from rlmesh import spaces

    # SPC-P6: Box(0.5, 3.7, ..., "int32") used to truncate to 0..3.
    with pytest.raises(ValueError, match="not an integer"):
        spaces.Box(0.5, 1.0, shape=[2], dtype=dtype)
    with pytest.raises(ValueError, match="not an integer"):
        spaces.Box(0, 3.7, shape=[2], dtype=dtype)
    with pytest.raises(ValueError, match="not an integer"):
        spaces.Box(0.5, float("inf"), shape=[2], dtype=dtype)
    # Integral floats are exact and accepted.
    assert spaces.Box(0.0, 1.0, shape=[2], dtype=dtype).high == 1


def test_box_contains_requires_the_exact_shape() -> None:
    from rlmesh import Tensor, spaces

    # SPC-P7: the element count alone no longer makes a member.
    box = spaces.Box(0.0, 1.0, shape=[2, 3])
    assert box.contains([[0.5] * 3] * 2)
    assert not box.contains([0.5] * 6)
    assert box.contains(Tensor(b"\0" * 24, [2, 3], "float32"))
    assert not box.contains(Tensor(b"\0" * 24, [6], "float32"))
    assert not box.contains([[0.5] * 3, [0.5] * 2])  # ragged
    # A bare number has shape (), so it is not a member of a (1,) Box.
    assert not spaces.Box(0.0, 1.0, shape=[1]).contains(0.5)
    assert spaces.Box(0.0, 1.0, shape=[1]).contains([0.5])
    nested = spaces.Dict({"pos": box})
    assert nested.contains({"pos": [[0.5] * 3] * 2})
    assert not nested.contains({"pos": [0.5] * 6})


def test_tensor_element_count_overflow_is_a_value_error() -> None:
    from rlmesh import Tensor

    # SPC-P9: was OverflowError.
    with pytest.raises(ValueError, match="overflow"):
        Tensor(b"", [2**62, 2**62], "float32")


def test_discrete_samples_at_signed_64_bit_boundary() -> None:
    from rlmesh import spaces

    for start in (-(2**63), 2**63 - 1):
        space = spaces.Discrete(1, start=start)
        assert space.sample() == start
        assert space.contains(start)


@pytest.mark.parametrize(
    ("low", "high", "dtype"), [(-1, 2**63, "int64"), (0, 2**64, "uint64")]
)
def test_box_refuses_bounds_outside_64_bit_ranges(low, high, dtype) -> None:
    from rlmesh import spaces

    with pytest.raises(ValueError, match="64-bit integer range"):
        spaces.Box(low, high, [1], dtype)


def test_box_contains_materializes_iterables_and_wrappers_once() -> None:
    from rlmesh import spaces

    class Array:
        calls = 0

        def tolist(self):
            self.calls += 1
            assert self.calls == 1
            return [0, 1]

    box = spaces.Box(0, 1, [2])
    assert box.contains(range(2))
    assert box.contains(x for x in range(2))
    assert box.contains(Array())
    assert not spaces.Box(0, 1, [1, 2]).contains(x for x in range(2))


def test_dict_contains_checks_box_shape_inside_namedtuple() -> None:
    from collections import namedtuple

    from rlmesh import spaces

    Child = namedtuple("Child", ["array"])
    space = spaces.Dict({"array": spaces.Box(0, 1, [2, 1])})
    assert not space.contains(Child([0, 1]))
    assert space.contains(Child([[0], [1]]))
