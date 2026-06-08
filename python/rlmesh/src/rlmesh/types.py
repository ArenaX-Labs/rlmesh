"""Public structural protocols and shared value aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, SupportsFloat, TypeAlias, TypeVar

from ._rlmesh import Tensor as _Tensor

PrimitiveValue: TypeAlias = None | bool | int | float | str | bytes
Value: TypeAlias = (
    PrimitiveValue | _Tensor | list["Value"] | tuple["Value", ...] | dict[str, "Value"]
)
Metadata: TypeAlias = Mapping[str, object]
InfoDict: TypeAlias = dict[str, object]

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
SpaceT = TypeVar("SpaceT", covariant=True)
BatchActionT = TypeVar("BatchActionT", contravariant=True)
EnvObsT = TypeVar("EnvObsT", covariant=True)
EnvActT = TypeVar("EnvActT")
VectorObsT = TypeVar("VectorObsT", covariant=True)
VectorActT = TypeVar("VectorActT", covariant=True)


class SpaceLike(Protocol[SpaceT]):
    """Structural protocol for RLMesh-compatible spaces."""

    def sample(self) -> SpaceT:
        """Return one valid sample from the space."""
        ...

    def contains(self, value: Any, /) -> bool:
        """Return whether ``value`` belongs to the space."""
        ...

    def seed(self, seed: int | None = None) -> int | list[int] | dict[str, int] | None:
        """Seed the space sampler."""
        ...


class EnvLike(Protocol[EnvObsT, EnvActT]):
    """Structural protocol for single environment."""

    @property
    def observation_space(self) -> SpaceLike[EnvObsT]:
        """Space describing reset and step observations."""
        ...

    @property
    def action_space(self) -> SpaceLike[EnvActT]:
        """Space describing accepted actions."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: InfoDict | None = None,
    ) -> tuple[EnvObsT, InfoDict]:
        """Reset the environment and return the initial observation."""
        ...

    def step(
        self,
        action: EnvActT,
    ) -> tuple[EnvObsT, SupportsFloat, bool, bool, InfoDict]:
        """Apply one action and return the Gymnasium-style step tuple."""
        ...

    def close(self) -> None:
        """Release environment resources."""
        ...


class VectorEnvLike(Protocol[BatchActionT, VectorObsT, VectorActT]):
    """Structural protocol for vectorized environments."""

    @property
    def num_envs(self) -> int:
        """Number of environment instances in the vector."""
        ...

    @property
    def single_observation_space(self) -> SpaceLike[VectorObsT]:
        """Observation space for one environment in the vector."""
        ...

    @property
    def single_action_space(self) -> SpaceLike[VectorActT]:
        """Action space for one environment in the vector."""
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: InfoDict | None = None,
    ) -> tuple[object, InfoDict]:
        """Reset all environments and return batched observations."""
        ...

    def step(
        self,
        actions: BatchActionT,
    ) -> tuple[object, object, object, object, InfoDict]:
        """Apply a batch of actions and return batched step values."""
        ...

    def close(self) -> None:
        """Release vector environment resources."""
        ...


__all__ = [
    "EnvLike",
    "InfoDict",
    "Metadata",
    "PrimitiveValue",
    "SpaceLike",
    "Value",
    "VectorEnvLike",
]
