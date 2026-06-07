"""Shared single-environment remote client base."""

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeVar, cast

from .._values import ValueAdapter
from ..specs import EnvContract, SpaceSpec
from ..types import Metadata, Value
from .endpoint import Transport, normalize_connect_address
from .viewer import EMPTY_METADATA, RenderPacket, ViewerMixin, ViewerProcess

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyEnvClient, ResetInfo, StepInfo

    from ..spaces import Space

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")


class RemoteEnvBase(ViewerMixin, Generic[ValueT, ActionT]):
    """Base class for backend-specific single-environment remote clients.

    Backend modules such as ``rlmesh.numpy`` and ``rlmesh.torch`` configure the
    value adapter. User code should normally instantiate those concrete
    backends instead of this base class.

    Args:
        address: Endpoint address such as ``"tcp://127.0.0.1:5555"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
    """

    _adapter: ClassVar[ValueAdapter]
    _address: str
    _observation_space_loaded: bool
    _action_space_loaded: bool
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
        try:
            from rlmesh._rlmesh import PyEnvClient
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        self._adapter.ensure_available()
        normalized_address = normalize_connect_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        self._client: PyEnvClient = PyEnvClient(normalized_address)
        self._address = self._client.address()
        self._env_contract: EnvContract = self._client.handshake()
        self._observation_space: Space | None = None
        self._action_space: Space | None = None
        self._observation_space_loaded = False
        self._action_space_loaded = False
        self._viewer: ViewerProcess | None = None
        self._viewer_warning_emitted = False

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
    def observation_space(self) -> Space | None:
        """Observation space loaded from the remote environment contract."""
        if not self._observation_space_loaded:
            self._observation_space = self._load_space("observation")
            self._observation_space_loaded = True
        return self._observation_space

    @property
    def action_space(self) -> Space | None:
        """Action space loaded from the remote environment contract."""
        if not self._action_space_loaded:
            self._action_space = self._load_space("action")
            self._action_space_loaded = True
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
    def observation_space_spec(self) -> SpaceSpec | None:
        """Native observation space spec reported by the endpoint."""
        return self._env_contract.observation_space

    @property
    def action_space_spec(self) -> SpaceSpec | None:
        """Native action space spec reported by the endpoint."""
        return self._env_contract.action_space

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
        return cast(ValueT, self._adapter.decode(obs)), info

    def step(self, action: ActionT) -> tuple[ValueT, float, bool, bool, StepInfo]:
        """Step the remote environment with one encoded action.

        Args:
            action: Action accepted by the remote environment action space.

        Returns:
            Observation, reward, terminated flag, truncated flag, and step info.
        """
        obs, reward, terminated, truncated, info = self._client.step(
            self._adapter.encode(action)
        )
        self._refresh_viewer(pace=True)
        return (
            cast(ValueT, self._adapter.decode(obs)),
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
            return cast(ValueT | None, self._adapter.decode(frame))
        return cast(
            ValueT | None,
            self._adapter.decode(self._client.render(env_index=env_index)),
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
            f"{type(self).__name__}(address={self._address!r}, "
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

    def _load_space(self, kind: Literal["observation", "action"]) -> Space | None:
        from ..spaces import space_from_spec

        spec = (
            self._env_contract.observation_space
            if kind == "observation"
            else self._env_contract.action_space
        )
        if spec is None:
            return None
        return space_from_spec(spec)


__all__ = ["ActionT", "RemoteEnvBase", "ValueT"]
