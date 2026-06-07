"""Named RLMesh-native spaces and optional Gymnasium adapters."""

from __future__ import annotations

import importlib
import string
from collections.abc import Iterable, Mapping
from collections.abc import Sequence as SequenceType
from types import MappingProxyType
from typing import Any, ClassVar, cast, final

from ._rlmesh import Space as _SpaceHandle
from ._rlmesh import (
    box_space_spec,
    dict_space_spec,
    discrete_space_spec,
    multi_binary_space_spec,
    multi_discrete_space_spec,
    text_space_spec,
    tuple_space_spec,
)
from .specs import SpaceSpec

EMPTY_SPACE_MAPPING: Mapping[str, object] = MappingProxyType({})


class Space:
    """Base wrapper for an RLMesh-native space."""

    __slots__: ClassVar[tuple[str, ...]] = ("_native", "_spec")

    def __init__(self, spec: SpaceSpec) -> None:
        self._spec: SpaceSpec = spec
        self._native: _SpaceHandle | None = None

    @property
    def spec(self) -> SpaceSpec:
        """Native space specification backing this wrapper."""
        return self._spec

    @property
    def kind(self) -> str:
        """Native RLMesh space kind."""
        return self._spec.kind

    @property
    def shape(self) -> list[int]:
        """Native shape for tensor-like spaces."""
        return self._spec.shape

    @property
    def dtype(self) -> str:
        """Element dtype reported by the native spec."""
        return self._spec.dtype

    def seed(self, seed: int | None = None) -> int | None:
        """Seed the native sampler for this space."""
        return self._native_space().seed(seed)

    def sample(self) -> object:
        """Sample a value from this space."""
        return self._native_space().sample()

    def contains(self, value: object) -> bool:
        """Return whether a value is contained in this space."""
        return self._native_space().contains(value)

    def to_gymnasium_space(self) -> object:
        """Convert this wrapper into a Gymnasium space."""
        return to_gymnasium_space(self)

    def _native_space(self) -> _SpaceHandle:
        native = self._native
        if native is None:
            native = self._spec.to_space()
            self._native = native
        return native

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        return _spec_to_dict(self._spec) == _spec_to_dict(other._spec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(kind={self.kind!r}, "
            f"shape={self.shape!r}, dtype={self.dtype!r})"
        )

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"{self.__class__.__name__!s} has no attribute {name!r}. "
            + "For Gymnasium compatibility, convert this space with "
            + "`rlmesh.spaces.to_gymnasium_space(...)`."
        )


@final
class Box(Space):
    """Continuous box space.

    Args:
        low: Lower bound, or an existing native ``SpaceSpec``.
        high: Upper bound when constructing a new spec.
        shape: Box shape when constructing a new spec.
        dtype: Element dtype name.
    """

    __slots__ = ("bounds_kind", "high", "low")
    bounds_kind: str | None
    low: object
    high: object

    def __init__(
        self,
        low: float | SpaceSpec,
        high: float | None = None,
        shape: SequenceType[int] | None = None,
        dtype: str = "float32",
    ) -> None:
        spec = (
            low
            if isinstance(low, SpaceSpec)
            else box_space_spec(
                float(low), _require_float(high, "high"), _shape(shape), dtype
            )
        )
        super().__init__(spec)
        details = _spec_details(spec)
        self.bounds_kind = cast(str | None, details.get("bounds_kind"))
        self.low = details.get("low")
        self.high = details.get("high")


@final
class Discrete(Space):
    """Discrete integer space.

    Args:
        n: Number of values, or an existing native ``SpaceSpec``.
        start: First integer value in the space.
        dtype: Integer dtype name.
    """

    __slots__ = ("n", "start")
    n: int
    start: int

    def __init__(
        self, n: int | SpaceSpec, start: int = 0, dtype: str = "int64"
    ) -> None:
        spec = n if isinstance(n, SpaceSpec) else discrete_space_spec(n, start, dtype)
        super().__init__(spec)
        details = _spec_details(spec)
        self.n = cast(int, details["n"])
        self.start = cast(int, details["start"])


@final
class MultiBinary(Space):
    """Binary vector or tensor space.

    Args:
        shape: Number of binary values, tensor shape, or native ``SpaceSpec``.
    """

    __slots__ = ("dims", "size")
    size: int | None
    dims: list[int] | None

    def __init__(self, shape: int | SequenceType[int] | SpaceSpec) -> None:
        spec = (
            shape
            if isinstance(shape, SpaceSpec)
            else multi_binary_space_spec(
                _shape([shape] if isinstance(shape, int) else shape)
            )
        )
        super().__init__(spec)
        details = _spec_details(spec)
        self.size = cast(int | None, details.get("size"))
        self.dims = cast(list[int] | None, details.get("dims"))


@final
class MultiDiscrete(Space):
    """Vector of discrete dimensions.

    Args:
        nvec: Per-dimension counts, or an existing native ``SpaceSpec``.
        dtype: Integer dtype name.
    """

    __slots__ = ("nvec",)
    nvec: list[int] | None

    def __init__(
        self, nvec: SequenceType[int] | SpaceSpec, dtype: str = "int64"
    ) -> None:
        spec = (
            nvec
            if isinstance(nvec, SpaceSpec)
            else multi_discrete_space_spec(list(nvec), dtype)
        )
        super().__init__(spec)
        details = _spec_details(spec)
        self.nvec = cast(list[int] | None, details.get("nvec"))


@final
class Text(Space):
    """Bounded text space.

    Args:
        max_length: Maximum string length, or an existing native ``SpaceSpec``.
        min_length: Minimum string length.
        charset: Optional allowed character set.
    """

    __slots__ = ("charset", "max_length", "min_length")
    min_length: int
    max_length: int
    charset: str

    def __init__(
        self,
        max_length: int | SpaceSpec,
        min_length: int = 1,
        charset: str | None = None,
    ) -> None:
        spec = (
            max_length
            if isinstance(max_length, SpaceSpec)
            else text_space_spec(max_length, min_length, charset)
        )
        super().__init__(spec)
        details = _spec_details(spec)
        self.min_length = cast(int, details["min_length"])
        self.max_length = cast(int, details["max_length"])
        self.charset = cast(str, details["charset"])


@final
class Dict(Space):
    """Mapping of named child spaces.

    Args:
        spaces: Mapping of child spaces, or an existing native ``SpaceSpec``.
    """

    __slots__ = ("spaces",)
    spaces: Mapping[str, Space]

    def __init__(self, spaces: Mapping[str, Space | SpaceSpec] | SpaceSpec) -> None:
        if isinstance(spaces, SpaceSpec):
            if spaces.kind != "dict":
                raise ValueError(f"expected dict SpaceSpec, got {spaces.kind!r}")
            spec = spaces
        else:
            entries: dict[str, object] = {key: child for key, child in spaces.items()}
            spec = dict_space_spec(entries)
        super().__init__(spec)
        details = _spec_details(spec)
        raw_spaces = cast(
            Mapping[str, SpaceSpec], details.get("spaces", EMPTY_SPACE_MAPPING)
        )
        self.spaces = MappingProxyType(
            {key: space_from_spec(child) for key, child in raw_spaces.items()}
        )


@final
class Tuple(Space):
    """Ordered tuple of child spaces.

    Args:
        spaces: Iterable of child spaces, or an existing native ``SpaceSpec``.
    """

    __slots__ = ("spaces",)
    spaces: tuple[Space, ...]

    def __init__(self, spaces: Iterable[Space | SpaceSpec] | SpaceSpec) -> None:
        if isinstance(spaces, SpaceSpec):
            if spaces.kind != "tuple":
                raise ValueError(f"expected tuple SpaceSpec, got {spaces.kind!r}")
            spec = spaces
        else:
            entries: list[object] = list(spaces)
            spec = tuple_space_spec(entries)
        super().__init__(spec)
        details = _spec_details(spec)
        raw_spaces = cast(list[SpaceSpec], details.get("spaces", []))
        self.spaces = tuple(space_from_spec(child) for child in raw_spaces)


_SPACE_BY_KIND: dict[str, type[Space]] = {
    "box": Box,
    "discrete": Discrete,
    "multi_binary": MultiBinary,
    "multi_discrete": MultiDiscrete,
    "text": Text,
    "dict": Dict,
    "tuple": Tuple,
}


def space_from_spec(spec: SpaceSpec) -> Space:
    """Create the named RLMesh space wrapper for a native spec.

    Args:
        spec: Native space specification.

    Returns:
        Matching RLMesh space wrapper.
    """
    kind = spec.kind
    cls = _SPACE_BY_KIND.get(kind)
    if cls is None:
        raise ValueError(f"unsupported RLMesh space kind: {kind}")
    return cls(spec)


def from_gymnasium_space(space: object) -> Space:
    """Convert a Gymnasium space into an RLMesh space wrapper.

    Args:
        space: Gymnasium or legacy Gym space.

    Returns:
        Matching RLMesh space wrapper.
    """
    gym_spaces = _gym_spaces()
    gym_space = cast(Any, space)

    if isinstance(space, gym_spaces.Box):
        return _box_from_gymnasium(gym_space)
    if isinstance(space, gym_spaces.Discrete):
        return Discrete(
            int(gym_space.n),
            start=int(getattr(gym_space, "start", 0)),
            dtype=str(getattr(gym_space, "dtype", "int64")),
        )
    if isinstance(space, gym_spaces.MultiBinary):
        return MultiBinary(_space_shape(gym_space.n))
    if isinstance(space, gym_spaces.MultiDiscrete):
        return _multi_discrete_from_gymnasium(gym_space)
    if hasattr(gym_spaces, "Text") and isinstance(space, gym_spaces.Text):
        return Text(
            int(gym_space.max_length),
            min_length=int(getattr(gym_space, "min_length", 1)),
            charset=_gymnasium_text_charset(gym_space),
        )
    if isinstance(space, gym_spaces.Dict):
        return Dict(
            {
                str(key): from_gymnasium_space(child)
                for key, child in gym_space.spaces.items()
            }
        )
    if isinstance(space, gym_spaces.Tuple):
        return Tuple(from_gymnasium_space(child) for child in gym_space.spaces)

    from ._rlmesh import space_spec_from_gym_space

    return space_from_spec(space_spec_from_gym_space(space))


def to_gymnasium_space(space: Space | SpaceSpec) -> object:
    """Convert an RLMesh space or native spec into a Gymnasium space.

    Args:
        space: RLMesh space wrapper or native space specification.

    Returns:
        Gymnasium space object.
    """
    spec = space.spec if isinstance(space, Space) else space
    gym_spaces = _gym_spaces()

    if spec.kind == "box":
        return _box_to_gymnasium(gym_spaces, spec)
    if spec.kind == "discrete":
        details = _spec_details(spec)
        n = cast(int, details["n"])
        start = cast(int, details.get("start", 0))
        return gym_spaces.Discrete(
            int(n),
            start=int(start),
        )
    if spec.kind == "multi_binary":
        details = _spec_details(spec)
        shape = details.get("dims", details.get("size", spec.shape))
        return gym_spaces.MultiBinary(shape)
    if spec.kind == "multi_discrete":
        details = _spec_details(spec)
        return gym_spaces.MultiDiscrete(details["nvec"])
    if spec.kind == "text":
        details = _spec_details(spec)
        max_length = cast(int, details["max_length"])
        min_length = cast(int, details.get("min_length", 1))
        charset = cast(str, details.get("charset", ""))
        return gym_spaces.Text(
            int(max_length),
            min_length=int(min_length),
            charset=charset or string.printable,
        )
    if spec.kind == "dict":
        details = _spec_details(spec)
        raw_spaces = cast(Mapping[str, SpaceSpec], details["spaces"])
        return gym_spaces.Dict(
            {key: to_gymnasium_space(child) for key, child in raw_spaces.items()}
        )
    if spec.kind == "tuple":
        details = _spec_details(spec)
        raw_spaces = cast(list[SpaceSpec], details["spaces"])
        return gym_spaces.Tuple(
            tuple(to_gymnasium_space(child) for child in raw_spaces)
        )
    return spec.to_gym_space()


def _gym_spaces() -> Any:
    last_error: ImportError | None = None
    for module_name in ("gymnasium", "gym"):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            last_error = exc
            continue
        return module.spaces
    raise ImportError(
        "Gymnasium space conversion requires gymnasium or gym. "
        "Install rlmesh[gymnasium]."
    ) from last_error


def _gymnasium_text_charset(space: Any) -> str:
    characters = getattr(space, "characters", None)
    if isinstance(characters, str):
        return characters

    character_list = getattr(space, "character_list", None)
    if character_list is not None:
        return "".join(str(character) for character in character_list)

    character_set = getattr(space, "character_set", None)
    if character_set is not None:
        return "".join(sorted(str(character) for character in character_set))

    charset = getattr(space, "charset", None)
    if charset is None:
        return ""
    if isinstance(charset, str):
        return charset
    return "".join(sorted(str(character) for character in charset))


def _box_from_gymnasium(space: Any) -> Box:
    import numpy as np

    low = np.asarray(space.low)
    high = np.asarray(space.high)
    if (
        low.size > 0
        and high.size > 0
        and np.all(low == low.flat[0])
        and np.all(high == high.flat[0])
    ):
        return Box(
            float(low.flat[0]),
            float(high.flat[0]),
            shape=_space_shape(space.shape),
            dtype=str(getattr(space, "dtype", "float32")),
        )

    from ._rlmesh import space_spec_from_gym_space

    return cast(Box, space_from_spec(space_spec_from_gym_space(space)))


def _box_to_gymnasium(gym_spaces: Any, spec: SpaceSpec) -> object:
    import numpy as np

    details = _spec_details(spec)
    shape = tuple(spec.shape)
    dtype = np.dtype(spec.dtype)
    bounds_kind = details.get("bounds_kind")
    if bounds_kind == "unbounded":
        low = np.full(shape, -np.inf, dtype=dtype)
        high = np.full(shape, np.inf, dtype=dtype)
    elif bounds_kind == "uniform":
        low = np.full(shape, details["low"], dtype=dtype)
        high = np.full(shape, details["high"], dtype=dtype)
    else:
        low = np.asarray(details["low"], dtype=dtype).reshape(shape)
        high = np.asarray(details["high"], dtype=dtype).reshape(shape)
    return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _multi_discrete_from_gymnasium(space: Any) -> MultiDiscrete:
    import numpy as np

    nvec = np.asarray(space.nvec)
    if nvec.ndim == 1:
        return MultiDiscrete([int(value) for value in nvec.tolist()])

    from ._rlmesh import space_spec_from_gym_space

    return cast(MultiDiscrete, space_from_spec(space_spec_from_gym_space(space)))


def _space_shape(value: object) -> list[int]:
    if isinstance(value, int):
        return [value]
    return [int(dim) for dim in cast(SequenceType[Any], value)]


def _shape(shape: SequenceType[int] | None) -> list[int]:
    if shape is None:
        raise TypeError("shape is required")
    return [int(dim) for dim in shape]


def _require_float(value: float | None, name: str) -> float:
    if value is None:
        raise TypeError(f"{name} is required")
    return float(value)


def _spec_details(spec: SpaceSpec) -> Mapping[str, object]:
    return cast(Mapping[str, object], spec._details())


def _spec_to_dict(spec: SpaceSpec) -> dict[str, object]:
    return spec._to_dict()


__all__ = [
    "Box",
    "Dict",
    "Discrete",
    "MultiBinary",
    "MultiDiscrete",
    "Space",
    "SpaceSpec",
    "Text",
    "Tuple",
    "from_gymnasium_space",
    "space_from_spec",
    "to_gymnasium_space",
]
