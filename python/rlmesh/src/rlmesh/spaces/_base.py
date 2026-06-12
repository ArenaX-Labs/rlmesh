from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, cast

from .._rlmesh import Space as _SpaceHandle
from ..specs import SpaceSpec
from ._utils import spec_to_dict

OutputT = TypeVar("OutputT")
NewOutputT = TypeVar("NewOutputT")


class SpaceBridge(Generic[OutputT]):
    """Convert space samples and inputs for a Python backend."""

    __slots__: ClassVar[tuple[str, ...]] = ("_input", "_sample")

    def __init__(
        self,
        sample: Callable[[object, SpaceSpec], OutputT],
        input: Callable[[object, SpaceSpec], object] | None = None,
    ) -> None:
        self._sample = sample
        self._input = input

    def sample(self, value: object, spec: SpaceSpec) -> OutputT:
        """Convert a native space sample into the backend value type."""
        return self._sample(value, spec)

    def input(self, value: object, spec: SpaceSpec) -> object:
        """Convert a backend value into the native input accepted by contains()."""
        if self._input is None:
            return value
        return self._input(value, spec)


class Space(Generic[OutputT]):
    """Base wrapper for an RLMesh-native space."""

    __slots__: ClassVar[tuple[str, ...]] = ("_bridge", "_native", "_spec")

    def __init__(self, spec: SpaceSpec) -> None:
        self._spec: SpaceSpec = spec
        self._native: _SpaceHandle | None = None
        self._bridge: SpaceBridge[OutputT] = cast(
            SpaceBridge[OutputT], _native_space_bridge()
        )

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
        return self._bridge.sample(self._native_space().sample(), self._spec)

    def contains(self, value: object) -> bool:
        """Return whether a value is contained in this space."""
        return self._native_space().contains(self._bridge.input(value, self._spec))

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

    def _with_bridge(self, bridge: SpaceBridge[NewOutputT]) -> Space[NewOutputT]:
        space = cast(Space[NewOutputT], self)
        space._bridge = bridge
        return space

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        return spec_to_dict(self._spec) == spec_to_dict(other._spec)

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


def _native_space_bridge() -> SpaceBridge[Any]:
    from ._sample import NATIVE_SPACE_BRIDGE

    return NATIVE_SPACE_BRIDGE
