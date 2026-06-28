"""Shared vector-environment remote client base."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from .._value_conversion import encode_framework_array_batch
from ..spaces import Space
from ..types import Metadata
from ._remote_base import ActionT, RemoteClientBase, ValueT

if TYPE_CHECKING:
    from rlmesh._rlmesh import ResetInfo, StepInfo


class RemoteVectorEnvBase(RemoteClientBase[ValueT, ActionT]):
    """Base class for backend-specific vector-environment remote clients.

    Backend modules such as ``rlmesh.numpy`` and ``rlmesh.torch`` configure the
    value bridge. User code should normally instantiate those concrete
    backends instead of this base class.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
    """

    _single_observation_space: Space[ValueT] | None = None
    _single_action_space: Space[ActionT] | None = None

    def _make_client(self, address: str, connect_timeout_seconds: float | None) -> Any:
        from .._load_native import load_native

        return load_native("PyVectorEnvClient")(
            address, connect_timeout_seconds=connect_timeout_seconds
        )

    def _adapt_metadata(self, metadata: Mapping[str, object]) -> Metadata:
        """Restore the Gymnasium ``AutoresetMode`` enum on the endpoint metadata.

        The wire codec degrades Gymnasium's ``AutoresetMode`` enum to its plain
        string value, but Gymnasium 1.x vector consumers (for example
        ``RecordEpisodeStatistics``) assert that ``metadata["autoreset_mode"]`` is
        an ``AutoresetMode`` instance. We restore the enum here so a
        Gymnasium-compliant training loop can read the server-side autoreset
        convention.
        """
        return _normalize_autoreset_mode(metadata)

    @property
    def num_envs(self) -> int:
        """Number of environment instances served by the endpoint."""
        return self._client.num_envs()

    @property
    def single_observation_space(self) -> Space[ValueT]:
        """Observation space for one environment in the vector."""
        if self._single_observation_space is None:
            self._single_observation_space = self._load_observation_space()
        return self._single_observation_space

    @property
    def single_action_space(self) -> Space[ActionT]:
        """Action space for one environment in the vector."""
        if self._single_action_space is None:
            self._single_action_space = self._load_action_space()
        return self._single_action_space

    @property
    def observation_space(self) -> Space[ValueT]:
        """Alias for `single_observation_space`."""
        return self.single_observation_space

    @property
    def action_space(self) -> Space[ActionT]:
        """Alias for `single_action_space`."""
        return self.single_action_space

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ValueT, ResetInfo]:
        """Reset all remote environments and decode the observations.

        Args:
            seed: Optional seed or per-environment seed list.
            options: Optional reset options forwarded to the vector environment.

        Returns:
            Batched decoded observations and reset info dictionary.
        """
        if isinstance(seed, list):
            seeds = seed
        elif seed is None:
            seeds = None
        else:
            seeds = [seed]
        obs, info = self._client.reset(seeds=seeds, options=options)
        return cast("ValueT", self._bridge.decode(obs)), info

    def step(self, actions: ActionT) -> tuple[ValueT, ValueT, ValueT, ValueT, StepInfo]:
        """Step all remote environments with a batch of actions.

        Args:
            actions: Batched actions accepted by the vector endpoint.

        Returns:
            Batched observations, rewards, terminations, truncations, and info.
        """
        obs, rewards, terminated, truncated, info = self._client.step(
            self._encode_actions(actions)
        )
        return (
            cast("ValueT", self._bridge.decode(obs)),
            cast("ValueT", self._bridge.decode(rewards)),
            cast("ValueT", self._bridge.decode(terminated)),
            cast("ValueT", self._bridge.decode(truncated)),
            info,
        )

    def _encode_actions(self, actions: ActionT) -> object:
        """Encode a batched action through the value bridge when possible.

        The value conversion module owns the framework-array batch heuristic;
        this client only supplies the action-space and lane context.
        """
        return encode_framework_array_batch(
            actions,
            bridge=self._bridge,
            space=self._space_spec("action"),
            num_envs=self.num_envs,
        )

    def __enter__(self) -> RemoteVectorEnvBase[ValueT, ActionT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.close()


def _normalize_autoreset_mode(metadata: Mapping[str, object]) -> Metadata:
    """Restore ``autoreset_mode`` to a Gymnasium ``AutoresetMode`` enum.

    The value is returned unchanged when the key is absent, when it is already
    an ``AutoresetMode``, when it does not match a known mode, or when
    Gymnasium is not installed.
    """
    mode = metadata.get("autoreset_mode")
    if mode is None or not isinstance(mode, str):
        return metadata
    try:
        from gymnasium.vector import AutoresetMode
    except ImportError:  # pragma: no cover - gymnasium optional
        return metadata
    try:
        enum_mode = AutoresetMode(mode)
    except ValueError:
        return metadata
    normalized = dict(metadata)
    normalized["autoreset_mode"] = enum_mode
    return normalized


__all__ = ["ActionT", "RemoteVectorEnvBase", "ValueT"]
