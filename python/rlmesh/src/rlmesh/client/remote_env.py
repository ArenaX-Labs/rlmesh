"""Shared single-environment remote client base."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, cast

from .._values import ValueBridge
from ..spaces import Space, SpaceBridge, space_from_spec
from ..specs import EnvContract, SpaceSpec
from ..types import Metadata, Value
from .endpoint import Transport, normalize_connect_address
from .viewer import EMPTY_METADATA, RenderPacket, ViewerMixin, ViewerProcess

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyEnvClient, ResetInfo, StepInfo

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")


class RemoteEnvBase(ViewerMixin, Generic[ValueT, ActionT]):
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

    _bridge: ClassVar[ValueBridge]
    _space_bridge: ClassVar[SpaceBridge[Any] | None] = None
    _address: str
    _viewer_warning_emitted: bool

    def __init__(
        self,
        address: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self._initialize(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
            connect_timeout_seconds=None,
        )

    @classmethod
    def _connect_for_sandbox(
        cls,
        address: str,
        *,
        connect_timeout_seconds: float,
    ) -> object:
        instance = cls.__new__(cls)
        instance._initialize(
            address,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        return instance

    def _initialize(
        self,
        address: str | None = None,
        *,
        host: str | None = None,
        port: int | None = None,
        path: str | None = None,
        transport: Transport | None = None,
        connect_timeout_seconds: float | None,
    ) -> None:
        try:
            from rlmesh._rlmesh import PyEnvClient
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        self._bridge.ensure_available()
        normalized_address = normalize_connect_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        self._client: PyEnvClient = PyEnvClient(
            normalized_address,
            connect_timeout_seconds=connect_timeout_seconds,
        )
        self._address = self._client.address()
        self._env_contract: EnvContract = self._client.handshake()
        if self._env_contract.num_envs > 1:
            num_envs = self._env_contract.num_envs
            try:
                self._client.close()
            finally:
                raise ValueError(
                    f"Endpoint {self._address!r} serves {num_envs} environments. "
                    "Use RemoteVectorEnv instead."
                )
        self._observation_space: Space[ValueT] | None = None
        self._action_space: Space[ActionT] | None = None
        self._viewer: ViewerProcess | None = None
        self._viewer_warning_emitted = False

    @property
    def address(self) -> str:
        """Endpoint address this client is connected to."""
        return self._address

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract returned by the endpoint handshake."""
        return self._env_contract

    @property
    def spec(self) -> EnvContract:
        """Alias for `env_contract`."""
        return self._env_contract

    @property
    def render_mode(self) -> str | None:
        """Configured render mode reported by the endpoint."""
        return self._env_contract.render_mode

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

    @property
    def metadata(self) -> Metadata:
        """Endpoint metadata reported by the environment contract."""
        metadata = self._env_contract.metadata
        if metadata is None:
            return EMPTY_METADATA
        return cast(Mapping[str, object], metadata)

    def _render_client(self) -> PyEnvClient:
        return self._client

    @property
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the endpoint."""
        return self._space_spec("observation")

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the endpoint."""
        return self._space_spec("action")

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
        self._refresh_viewer()
        return cast(ValueT, self._bridge.decode(obs)), info

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
        self._refresh_viewer(pace=True)
        return (
            cast(ValueT, self._bridge.decode(obs)),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self, *, env_index: int = 0) -> ValueT | None:
        """Render a frame from the remote environment.

        Args:
            env_index: Environment index to render. Single environments use
                ``0``.

        Returns:
            A decoded render frame, or ``None`` when the environment has no frame.
        """
        if self._viewer is not None and self._viewer.env_index == env_index:
            frame, packet = cast(
                tuple[Value | None, RenderPacket],
                self._client.render_bundle(env_index=env_index),
            )
            self._push_viewer_packet(packet)
            return cast(ValueT | None, self._bridge.decode(frame))
        return cast(
            ValueT | None,
            self._bridge.decode(self._client.render(env_index=env_index)),
        )

    def close(self) -> None:
        """Detach this client from the remote endpoint."""
        self._shutdown_viewer()
        self._client.close()

    def shutdown(self, reason: str = "owner shutdown") -> bool:
        """Request owner-level shutdown of the remote environment endpoint."""
        self._shutdown_viewer()
        return bool(self._client.shutdown(reason))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(address={self.address!r}, "
            f"env_id={self._env_contract.id!r})"
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

    def _space_spec(self, kind: Literal["observation", "action"]) -> SpaceSpec:
        return (
            self._env_contract.observation_space
            if kind == "observation"
            else self._env_contract.action_space
        )

    def _load_observation_space(self) -> Space[ValueT]:
        spec = self._space_spec("observation")
        bridge = self._space_bridge
        if bridge is None:
            return cast(Space[ValueT], space_from_spec(spec))
        return cast(Space[ValueT], space_from_spec(spec, bridge=bridge))

    def _load_action_space(self) -> Space[ActionT]:
        spec = self._space_spec("action")
        bridge = self._space_bridge
        if bridge is None:
            return cast(Space[ActionT], space_from_spec(spec))
        return cast(Space[ActionT], space_from_spec(spec, bridge=bridge))


__all__ = ["ActionT", "RemoteEnvBase", "ValueT"]
