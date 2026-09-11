"""Public structural protocols and shared value aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    SupportsFloat,
    TypeAlias,
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import TypeVar as _DefaultTypeVar

    from ._models._adapter_mode import (
        _NoAdapter,  # pyright: ignore[reportPrivateUsage]
    )
    from ._models._view import View as _View
    from ._rlmesh import Tensor as _Tensor
    from .adapters import ModelSpec as _ModelSpec
    from .specs import EnvContract as _EnvContract

PrimitiveValue: TypeAlias = None | bool | int | float | str | bytes
Value: TypeAlias = Union[
    PrimitiveValue,
    "_Tensor",
    list["Value"],
    tuple["Value", ...],
    dict[str, "Value"],
]
Metadata: TypeAlias = Mapping[str, object]
"""SDK-validated metadata mapping (e.g. adapter tags, describe envelopes).

Values are typed ``object`` on purpose: the SDK owns this surface and validates
it, so consumers narrow explicitly instead of trusting an ``Any``. Contrast
:data:`InfoDict`, the user-owned grab-bag."""

InfoDict: TypeAlias = dict[str, Any]
"""User-owned diagnostic grab-bag returned by env ``reset``/``step`` (the
gymnasium ``info`` norm).

Values are typed ``Any`` on purpose: the env author owns what goes in, the SDK
never validates it, so strict-mode consumers can write ``info["success"] > 0.5``
without a cast. Contrast :data:`Metadata`, which the SDK validates and therefore
types as ``object``."""


class PredictContext(TypedDict):
    """What a predict corner is told about the episode it is predicting for.

    Delivered as a trailing ``context`` argument, and only to a corner whose
    signature declares one -- so it is additive. A model that interleaves
    episodes keys everything it remembers by ``episode_id``; ``state`` is the
    slot to keep it in, so it does not need its own bookkeeping.

    Attributes:
        episode_id: The runtime-minted id of this episode (UUIDv7, never
            repeats). ``""`` for an anonymous lane with no identity.
        episode_seed: The seed this episode was reset with, or ``None`` when it
            was not explicitly seeded.
        predict_index: How many times this episode has been predicted for
            already -- 0 on its first predict. The re-plan ordinal, not the env
            step count (a chunked model re-plans once per chunk).
        predict_seed: A reproducible per-predict seed mixed from
            ``episode_seed`` and ``predict_index``; ``None`` when the episode
            carries no seed. Seed a stochastic head with this rather than once
            per episode: interleaved episodes advance the same global
            generators, so seeding once cannot reproduce.
        state: A dict that lives as long as the episode does, for whatever the
            model wants to carry across its predicts. Dropped when the episode
            ends.
    """

    episode_id: str
    episode_seed: int | None
    predict_index: int
    predict_seed: int | None
    state: dict[str, Any]


BatchPredictContext: TypeAlias = list[PredictContext]
"""One :class:`PredictContext` per row of a batched corner's batch, in row order.

The batched corners (``predict_batch`` / ``predict_chunk_batch``) run one forward
over lanes that may belong to independent episodes, so identity is per row --
never one context for the call.
"""

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
SpaceT = TypeVar("SpaceT", covariant=True)
EnvObsT = TypeVar("EnvObsT", covariant=True)
EnvActT = TypeVar("EnvActT")

if TYPE_CHECKING:
    BatchActionT = _DefaultTypeVar("BatchActionT", contravariant=True, default=Any)
    VectorObsT = _DefaultTypeVar("VectorObsT", covariant=True, default=Any)
    VectorActT = _DefaultTypeVar("VectorActT", covariant=True, default=Any)
    BatchObsT = _DefaultTypeVar("BatchObsT", covariant=True, default=Any)
    BatchArrayT = _DefaultTypeVar("BatchArrayT", covariant=True, default=Any)
else:
    BatchActionT = TypeVar("BatchActionT", contravariant=True)
    VectorObsT = TypeVar("VectorObsT", covariant=True)
    VectorActT = TypeVar("VectorActT", covariant=True)
    BatchObsT = TypeVar("BatchObsT", covariant=True)
    BatchArrayT = TypeVar("BatchArrayT", covariant=True)


class HasAddress(Protocol):
    """Anything exposing a dialable endpoint ``address`` (e.g. a server handle)."""

    @property
    def address(self) -> str:
        """The dialable endpoint address."""
        ...


class HasEnvContract(Protocol):
    """An env client exposing a published ``env_contract`` (e.g. a served-env handle)."""

    @property
    def env_contract(self) -> _EnvContract:
        """The env's published :class:`~rlmesh.specs.EnvContract`."""
        ...


class SupportsResetStep(Protocol):
    """A live env object: anything with ``reset`` and ``step`` callables."""

    @property
    def reset(self) -> Callable[..., Any]:
        """The env's reset callable."""
        ...

    @property
    def step(self) -> Callable[..., Any]:
        """The env's step callable."""
        ...


class SupportsMake(Protocol):
    """An env factory: anything with a ``make`` callable (e.g. :class:`~rlmesh.EnvFactory`)."""

    @property
    def make(self) -> Callable[..., Any]:
        """The factory's make callable."""
        ...


LocalEnvTarget: TypeAlias = Union[
    "SupportsResetStep", "SupportsMake", "HasAddress", str
]
"""What the local drive paths (``run()``/``session()``/``Model.run``) accept as an
env: a live env object, an :class:`~rlmesh.EnvFactory`, an object exposing an
``address``, or an address string to dial. Matched structurally, exactly like the
runtime's own checks. A contract-only handle (``env_contract`` without
``reset``/``step``) cannot be stepped locally, so it is deliberately absent."""

EnvTarget: TypeAlias = Union["LocalEnvTarget", "HasEnvContract"]
"""What a served-model ``session()`` (:class:`~rlmesh.RemoteModel` /
:class:`~rlmesh.SandboxModel`) accepts as an env: everything in
:data:`LocalEnvTarget`, plus an env client exposing a published ``env_contract``
(the contract is sent to the served model at bind)."""

SpecArg: TypeAlias = Union["_ModelSpec", "_NoAdapter", None]
"""A model's adapter spec argument: a :class:`~rlmesh.adapters.ModelSpec`,
:data:`rlmesh.NO_ADAPTER` to opt out on a tagged env, or ``None``."""

ViewArg: TypeAlias = Union["_View", str, bool, None]
"""The ``view=`` argument: a :class:`~rlmesh.View`, a shorthand string such as
``"terminal"`` or ``"http:9000"``, ``True`` for the default viewer, or ``None``."""


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
    """Structural protocol for a single environment."""

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


if TYPE_CHECKING:

    class VectorEnvLike(
        Protocol[BatchActionT, VectorObsT, VectorActT, BatchObsT, BatchArrayT]
    ):
        """Structural protocol for vectorized environments.

        Parametrized like :class:`EnvLike`, over the batched shapes: ``BatchActionT``
        is the batched action ``step`` accepts, ``VectorObsT``/``VectorActT`` are the
        per-instance space sample types, ``BatchObsT`` is the batched observation
        ``reset``/``step`` return, and ``BatchArrayT`` is the batched
        reward/termination/truncation array (one type for all three, the gymnasium
        vector norm). The batched parameters default to ``Any``, so an
        unparametrized ``VectorEnvLike`` (or the historical three-parameter
        spelling) stays ceremony-free.
        """

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
        ) -> tuple[BatchObsT, InfoDict]:
            """Reset all environments and return batched observations."""
            ...

        def step(
            self,
            actions: BatchActionT,
        ) -> tuple[BatchObsT, BatchArrayT, BatchArrayT, BatchArrayT, InfoDict]:
            """Apply a batch of actions and return batched step values."""
            ...

        def close(self) -> None:
            """Release vector environment resources."""
            ...

else:

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
    "BatchPredictContext",
    "EnvLike",
    "EnvTarget",
    "HasAddress",
    "HasEnvContract",
    "InfoDict",
    "LocalEnvTarget",
    "Metadata",
    "PredictContext",
    "PrimitiveValue",
    "SpaceLike",
    "SpecArg",
    "SupportsMake",
    "SupportsResetStep",
    "Value",
    "VectorEnvLike",
    "ViewArg",
]
