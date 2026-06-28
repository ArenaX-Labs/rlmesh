"""Shared remote-client base for single- and vector-environment clients.

Holds the connection/handshake/lifecycle and the space accessors common to
:class:`rlmesh._client._remote_env.RemoteEnvBase` and
:class:`rlmesh._client._remote_vector_env.RemoteVectorEnvBase`. The arity- and
backend-specific pieces -- which native client to build (``_make_client``),
post-handshake validation (``_post_handshake``), metadata adaptation
(``_adapt_metadata``), and the step/reset shapes + space-cache attributes -- are
subclass hooks/overrides. Not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

from .._value_conversion import ValueBridge
from ..spaces import Space, space_from_spec
from ..specs import EnvContract, SpaceSpec
from ..types import Metadata
from ._endpoint import Transport, normalize_connect_address
from ._metadata import EMPTY_METADATA

ValueT = TypeVar("ValueT")
ActionT = TypeVar("ActionT")


class RemoteClientBase(Generic[ValueT, ActionT]):
    """Connection, handshake, and lifecycle shared by remote env clients.

    Backend modules such as ``rlmesh.numpy`` and ``rlmesh.torch`` configure the
    value bridge on the concrete subclass. User code should instantiate those
    concrete backends, never this base.
    """

    _bridge: ClassVar[ValueBridge]
    _address: str
    _client: Any

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
        self._bridge.ensure_available()
        normalized_address = normalize_connect_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        self._client = self._make_client(normalized_address, connect_timeout_seconds)
        self._address = self._client.address()
        self._env_contract: EnvContract = self._client.handshake()
        self._post_handshake()

    # --- arity / backend hooks -------------------------------------------------
    def _make_client(self, address: str, connect_timeout_seconds: float | None) -> Any:
        """Build and return the native client. Overridden per arity."""
        raise NotImplementedError

    def _post_handshake(self) -> None:
        """Validate the handshake. Default: accept any environment count."""

    def _adapt_metadata(self, metadata: Mapping[str, object]) -> Metadata:
        """Post-process endpoint metadata. Default: return it unchanged."""
        return metadata

    # --- shared properties -----------------------------------------------------
    @property
    def address(self) -> str:
        """Endpoint address this client is connected to."""
        return self._address

    @property
    def env_id(self) -> str:
        """This connection's container id (UUIDv7).

        A stable correlation identity, distinct from the human env name
        (`env_contract.id`).
        """
        return self._client.env_id()

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
    def metadata(self) -> Metadata:
        """Endpoint metadata reported by the environment contract."""
        metadata = self._env_contract.metadata
        if metadata is None:
            return EMPTY_METADATA
        return self._adapt_metadata(cast("Mapping[str, object]", metadata))

    @property
    def observation_space_spec(self) -> SpaceSpec:
        """Native observation space spec reported by the endpoint."""
        return self._space_spec("observation")

    @property
    def action_space_spec(self) -> SpaceSpec:
        """Native action space spec reported by the endpoint."""
        return self._space_spec("action")

    # --- shared methods --------------------------------------------------------
    def render(self, *, env_index: int = 0) -> ValueT | None:
        """Render a frame from the remote environment.

        Args:
            env_index: Environment index to render. Single environments use ``0``.

        Returns:
            A decoded render frame, or ``None`` when the environment has no frame.
        """
        return cast(
            "ValueT | None",
            self._bridge.decode(self._client.render(env_index=env_index)),
        )

    def close(self) -> None:
        """Detach this client from the remote endpoint."""
        self._client.close()

    def shutdown(self, reason: str = "owner shutdown") -> bool:
        """Request owner-level shutdown of the remote environment endpoint."""
        return bool(self._client.shutdown(reason))

    def _space_spec(self, kind: Literal["observation", "action"]) -> SpaceSpec:
        return (
            self._env_contract.observation_space
            if kind == "observation"
            else self._env_contract.action_space
        )

    def _load_observation_space(self) -> Space[ValueT]:
        spec = self._space_spec("observation")
        return cast("Space[ValueT]", space_from_spec(spec, bridge=self._bridge))

    def _load_action_space(self) -> Space[ActionT]:
        spec = self._space_spec("action")
        return cast("Space[ActionT]", space_from_spec(spec, bridge=self._bridge))
