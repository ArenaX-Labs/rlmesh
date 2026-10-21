"""Shared remote-client base for single- and vector-environment clients.

Holds the connection/handshake/lifecycle and the space accessors common to
:class:`rlmesh._client._remote_env.RemoteEnvBase` and
:class:`rlmesh._client._remote_vector_env.RemoteVectorEnvBase`. The arity- and
backend-specific pieces -- which native client to build (``_make_client``),
metadata adaptation (``_adapt_metadata``), and the step/reset shapes +
space-cache attributes -- are subclass hooks/overrides. The native client dials,
handshakes and checks the endpoint's arity in its constructor. Not part of the
public API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

from .._editions import resolve_workflow_edition
from .._value_conversion import ValueBridge
from ..spaces import Space, space_from_spec
from ..specs import EnvContract, SpaceSpec
from ..types import Metadata
from . import _gym
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
    #: Whether this client is a Gymnasium env: with gymnasium installed, it
    #: exposes the env's own Gymnasium spaces and info; the NumPy backend
    #: also takes Gymnasium's value types (see ``_gym_values``).
    _gym: ClassVar[bool] = False
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
        connect_timeout_seconds: float | None = None,
        request_timeout_seconds: float | None = None,
        workflow_edition: str | None = None,
        api_key: str | None = None,
        tls_ca_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self._initialize(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
            connect_timeout_seconds=connect_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            workflow_edition=workflow_edition,
            api_key=api_key,
            tls_ca_file=tls_ca_file,
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
        request_timeout_seconds: float | None = None,
        workflow_edition: str | None = None,
        api_key: str | None = None,
        tls_ca_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self._bridge.ensure_available()
        # This client is the runtime tier of its session, so it declares an
        # edition like any other participant: the explicit keyword, else the
        # process / project surfaces (see :mod:`rlmesh._editions`).
        workflow_edition = resolve_workflow_edition(call=workflow_edition)
        normalized_address = normalize_connect_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        # Dials, handshakes and checks the arity; a dial failure already
        # reads "could not connect to an env at ...".
        # Only the credentials given travel, as keywords.
        credentials: dict[str, str] = {}
        if api_key is not None:
            credentials["api_key"] = api_key
        if tls_ca_file is not None:
            credentials["tls_ca_file"] = os.fspath(tls_ca_file)
        self._client = self._make_client(
            normalized_address,
            connect_timeout_seconds,
            request_timeout_seconds,
            workflow_edition,
            **credentials,
        )
        self._address = self._client.address()
        # A local conversion of the handshake's contract; no I/O.
        self._env_contract: EnvContract = self._client.handshake()
        # Accessors read state outside the client's per-call borrow, so they
        # work while another thread is in a call.
        self._env_id: str = self._client.env_id()
        self._edition_cell = self._client.edition_cell()

    # --- arity / backend hooks -------------------------------------------------
    def _make_client(
        self,
        address: str,
        connect_timeout_seconds: float | None,
        request_timeout_seconds: float | None,
        workflow_edition: str | None,
        **credentials: str,
    ) -> Any:
        """Build and return the native client. Overridden per arity.

        ``credentials`` carries ``api_key`` and ``tls_ca_file`` when given.
        """
        raise NotImplementedError

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
        return self._env_id

    @property
    def selected_workflow_edition(self) -> str:
        """The workflow edition this connection negotiated with the env.

        The highest edition both sides can drive, at or below whatever either
        declared, in the exact spelling they agreed on -- so a prerelease build
        reports its cohort, not the bare sealed base. Named apart from the
        ``workflow_edition`` *declaration* surfaces on purpose: this is what was
        agreed, not what this side asked for.
        """
        return cast("str", self._edition_cell.get())

    def _session_offer(self) -> tuple[list[str], str | None]:
        """The env's handshake offer ``(CAN, WANT)``, for a served-model session's floor."""
        return cast("tuple[list[str], str | None]", self._client.session_offer())

    def _pin_workflow_edition(self, selected_workflow_edition: str) -> None:
        """Pin the env to a served-model session's floor before its first reset."""
        self._client.pin_workflow_edition(selected_workflow_edition)

    @property
    def env_contract(self) -> EnvContract:
        """Environment contract returned by the endpoint handshake."""
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

    @property
    def _gym_active(self) -> bool:
        return type(self)._gym and _gym.gym is not None

    @property
    def _gym_values(self) -> bool:
        """Whether values convert to Gymnasium's NumPy types; torch and jax stay tensors."""
        return self._gym_active and getattr(self._bridge, "name", None) == "numpy"

    def _load_observation_space(self) -> Space[ValueT]:
        spec = self._space_spec("observation")
        if self._gym_active:
            return cast("Space[ValueT]", _gym.gym_space(spec))
        return cast("Space[ValueT]", space_from_spec(spec, bridge=self._bridge))

    def _load_action_space(self) -> Space[ActionT]:
        spec = self._space_spec("action")
        if self._gym_active:
            return cast("Space[ActionT]", _gym.gym_space(spec))
        return cast("Space[ActionT]", space_from_spec(spec, bridge=self._bridge))
