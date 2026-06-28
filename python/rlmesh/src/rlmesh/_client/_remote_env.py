"""Shared single-environment remote client base."""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

from ..spaces import Space
from ._remote_base import ActionT, RemoteClientBase, ValueT

if TYPE_CHECKING:
    from rlmesh._rlmesh import ResetInfo, StepInfo


class RemoteEnvBase(RemoteClientBase[ValueT, ActionT]):
    """Base class for backend-specific single-environment remote clients.

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

    _observation_space: Space[ValueT] | None = None
    _action_space: Space[ActionT] | None = None

    def _make_client(self, address: str, connect_timeout_seconds: float | None) -> Any:
        from .._load_native import load_native

        return load_native("PyEnvClient")(
            address, connect_timeout_seconds=connect_timeout_seconds
        )

    def _post_handshake(self) -> None:
        if self._env_contract.num_envs > 1:
            num_envs = self._env_contract.num_envs
            try:
                self._client.close()
            finally:
                raise ValueError(
                    f"Endpoint {self._address!r} serves {num_envs} environments. "
                    "Use RemoteVectorEnv instead."
                )

    @property
    def observation_space(self) -> Space[ValueT]:
        """Observation space loaded from the remote environment contract."""
        if self._observation_space is None:
            self._observation_space = self._load_observation_space()
        return self._observation_space

    @property
    def action_space(self) -> Space[ActionT]:
        """Action space loaded from the remote environment contract."""
        if self._action_space is None:
            self._action_space = self._load_action_space()
        return self._action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[ValueT, ResetInfo]:
        """Reset the remote environment and decode the observation.

        Args:
            seed: Optional environment reset seed.
            options: Optional reset options forwarded to the environment.

        Returns:
            A decoded observation and reset info dictionary.
        """
        seeds = [seed] if seed is not None else None
        obs, info = self._client.reset(seeds=seeds, options=options)
        return cast("ValueT", self._bridge.decode(obs)), info

    def step(self, action: ActionT) -> tuple[ValueT, float, bool, bool, StepInfo]:
        """Step the remote environment with one encoded action.

        Args:
            action: Action accepted by the remote environment action space.

        Returns:
            Observation, reward, terminated flag, truncated flag, and step info.
        """
        obs, reward, terminated, truncated, info = self._client.step(
            self._bridge.encode(action)
        )
        return (
            cast("ValueT", self._bridge.decode(obs)),
            reward,
            terminated,
            truncated,
            info,
        )

    def __enter__(self) -> RemoteEnvBase[ValueT, ActionT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.close()


__all__ = ["ActionT", "RemoteEnvBase", "ValueT"]
