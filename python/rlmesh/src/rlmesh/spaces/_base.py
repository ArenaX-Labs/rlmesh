from __future__ import annotations

from typing import ClassVar, Generic, TypeVar, cast

from .._rlmesh import Space as _SpaceHandle
from .._value_conversion import ValueBridge
from ..specs import SpaceSpec
from ..types import Value
from ._internals import spec_to_dict

OutputT = TypeVar("OutputT")
NewOutputT = TypeVar("NewOutputT")


class Space(Generic[OutputT]):
    """Base wrapper for an RLMesh-native space."""

    __slots__: ClassVar[tuple[str, ...]] = ("_bridge", "_native", "_spec")

    def __init__(self, spec: SpaceSpec) -> None:
        self._spec: SpaceSpec = spec
        self._native: _SpaceHandle | None = None
        self._bridge: ValueBridge = _native_value_bridge()

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

    def sample(self) -> OutputT:
        """Sample a value from this space."""
        return cast(
            OutputT, self._bridge.decode(cast("Value", self._native_space().sample()))
        )

    def contains(self, value: object) -> bool:
        """Return whether a value is contained in this space."""
        return self._native_space().contains(self._bridge.encode(value))

    def to_gymnasium_space(self) -> object:
        """Convert this wrapper into a Gymnasium space."""
        from ._conversion import to_gymnasium_space

        return to_gymnasium_space(self)

    def _native_space(self) -> _SpaceHandle:
        native = self._native
        if native is None:
            native = self._spec.to_space()
            self._native = native
        return native

    def _with_bridge(self, bridge: ValueBridge) -> Space[NewOutputT]:
        space = cast(Space[NewOutputT], self)
        space._bridge = bridge
        return space

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        return spec_to_dict(self._spec) == spec_to_dict(other._spec)

    def __repr__(self) -> str:
        # Gymnasium-style rendering from the native spec's Display, e.g.
        # `Box(-1.0, 1.0, (3,), float32)` / `Discrete(4, start=-1)`.
        return repr(self._spec)

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"{self.__class__.__name__!s} has no attribute {name!r}. "
            + "For Gymnasium compatibility, convert this space with "
            + "`rlmesh.spaces.to_gymnasium_space(...)`."
        )


def _native_value_bridge() -> ValueBridge:
    from ._internals import NATIVE_VALUE_BRIDGE

    return NATIVE_VALUE_BRIDGE
