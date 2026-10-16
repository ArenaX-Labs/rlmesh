"""Model client: an env-agnostic served-policy handle.

The model-side mirror of :mod:`._remote_env`. A :class:`RemoteModelBase` is an
env-agnostic handle to a served model (policy); :meth:`RemoteModelBase.session`
binds it to one env -- sending that env's contract (and adapter tags) so the served
model resolves its adapter -- and returns the neutral :class:`rlmesh.Session` you
drive with ``reset`` / ``predict`` / ``step``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from .._value_conversion import ValueBridge
from ..types import EnvTarget, ViewArg
from ._endpoint import Transport, normalize_connect_address

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._models._eval import Session
    from ..specs import EnvContract

logger = logging.getLogger(__name__)

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class RemoteModelBase(Generic[ObsT, ActT]):
    """Env-agnostic handle to a served model (policy).

    Bind it to an env with :meth:`session` to get a :class:`rlmesh.Session`. The handle
    carries no env contract, so one handle drives many envs: ``model.session(env_a)``
    and ``model.session(env_b)`` are independent sessions off the same policy.

    Args:
        address: Model endpoint address such as ``"tcp://127.0.0.1:5555"``,
            ``"127.0.0.1:5555"``, ``"https://models.example"`` (TLS, port 443
            by default) or ``"unix:///tmp/model.sock"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        connect_timeout_seconds: Optional bound on each session's dial to the
            model server; ``None`` (default) waits indefinitely.
        request_timeout_seconds: Optional per-request bound applied to every
            predict/close RPC of sessions off this handle; ``None`` (default)
            waits indefinitely.
        api_key: Key sent as ``authorization: Bearer <key>`` on every model
            request. ``None`` (default) reads ``RLMESH_API_KEY`` for an
            ``https://`` address only; ``""`` sends none. Never shown in
            errors or repr.
        tls_ca_file: PEM bundle an ``https://`` dial trusts instead of the
            system roots (like ``curl --cacert``).
    """

    _bridge: ClassVar[ValueBridge]

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
        api_key: str | None = None,
        tls_ca_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self._bridge.ensure_available()
        self._address = normalize_connect_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        from .._load_native import load_native

        ca_file = None if tls_ca_file is None else os.fspath(tls_ca_file)
        # The address must parse, the timeouts be non-negative and finite, the
        # key fit a header and the CA bundle load here, as RemoteEnv checks
        # them, not at the first session(). No network I/O.
        load_native("check_remote_model_options")(
            self._address,
            connect_timeout_seconds=connect_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
            api_key=api_key,
            tls_ca_file=ca_file,
        )
        self._connect_timeout_seconds = connect_timeout_seconds
        self._request_timeout_seconds = request_timeout_seconds
        # Held behind a redacting wrapper: never part of repr, vars() or errors.
        self._api_key = _Secret(api_key)
        self._tls_ca_file = ca_file

    @property
    def address(self) -> str:
        """Model endpoint address this handle dials."""
        return self._address

    def session(
        self,
        env: EnvTarget,
        *,
        instruction: str | None = None,
        close_env: bool = False,
        trust_entrypoints: bool | None = None,
        execution_horizon: int = 1,
        view: ViewArg = None,
        workflow_edition: str | None = None,
    ) -> Session[Any, Any]:
        """Bind this served policy to ``env`` and return a :class:`rlmesh.Session`.

        ``env`` is an env client (e.g. ``RemoteEnv``/``SandboxEnv``) exposing an
        ``env_contract``, an address to dial, or a plain local env (or an
        :class:`~rlmesh.EnvFactory`, whose env the session builds and closes),
        stepped on the caller's thread with no EnvServer. The env's contract --
        including its adapter tags -- is sent to the model server, which
        resolves its adapter for this env. A local vector env is a TypeError:
        a served model drives a single env. ``instruction``
        and ``trust_entrypoints`` apply to local models and are rejected here (the
        served model owns its adapter). ``view`` opts into the built-in live viewer
        over this client's own env loop (see :func:`rlmesh.run`).

        ``execution_horizon`` (> 1) opts a chunk-capable served model into action
        chunking: the model emits its native chunk, and this client replays a prefix
        of it one action per step open-loop (skipping the RPC), re-planning every
        ``execution_horizon`` steps. The model must define ``predict_chunk``;
        otherwise it re-plans every step.

        ``workflow_edition`` declares this runtime's edition for the session,
        ahead of the process and project declarations. It also overrides the
        runtime declaration used to connect the env handle; the env server's
        own declaration still constrains the session floor.
        """
        if instruction is not None:
            raise ValueError(
                "served models take the env's instruction; instruction= is not "
                "supported for RemoteModel sessions"
            )
        if trust_entrypoints is not None:
            raise ValueError(
                "trust_entrypoints= applies to local model sources; it is not "
                "supported for RemoteModel sessions"
            )
        from .._editions import resolve_workflow_edition
        from .._load_native import load_native
        from .._models._eval import check_execution_horizon
        from .._models._view import resolve_view

        # What the options alone decide fails before the env or model is dialed.
        check_execution_horizon(execution_horizon, None)
        view = resolve_view(view)

        # This handle is the runtime tier of the session it opens, so the model
        # leg carries the call, process, or project declaration. The env's
        # own offer still constrains the three-way floor.
        workflow_edition = resolve_workflow_edition(call=workflow_edition)
        from .._models._connect import (
            _native_contract,  # pyright: ignore[reportPrivateUsage]
            adapter_env_bridge,
            classify_env_target,
            factory_env,
        )

        # An address or a handle publishing a contract keeps the served path;
        # anything else classifies as the local loops do.
        kind = (
            "handle"
            if isinstance(env, str) or _native_contract(env) is not None
            else classify_env_target(env)[0]
        )
        bridge = None
        if kind in ("env", "factory"):
            # A local env is stepped on this thread; the model is bound with
            # the contract an EnvServer would publish for it. A factory's env
            # is built now and is the session's own.
            env, owns_env = (
                (factory_env(env), True) if kind == "factory" else (env, False)
            )
            try:
                contract = load_native("local_env_contract")(env)
            except BaseException:
                if owns_env:
                    close_dialed_env(env)
                raise
            # A raw local env hands out its own arrays (numpy for gymnasium).
            bridge = adapter_env_bridge(env)
            env_offer = None
        else:
            # The call's declaration is also the dialed env's handshake declaration.
            env, owns_env = dial_env_address(env, workflow_edition)
            contract = None
            env_offer = None
        try:
            if contract is None:
                contract = env_contract_of(env)
                env_offer = env_session_offer(env)
            # A dial failure already reads "could not connect to a model at ...".
            client = load_native("PyModelClient")(
                self._address,
                contract,
                execution_horizon,
                connect_timeout_seconds=self._connect_timeout_seconds,
                request_timeout_seconds=self._request_timeout_seconds,
                workflow_edition=workflow_edition,
                env_offer=env_offer,
                api_key=self._api_key.value,
                tls_ca_file=self._tls_ca_file,
            )
        except BaseException:
            # An env dialed or built for this call is released before the
            # error goes up, not left to the garbage collector.
            if owns_env:
                close_dialed_env(env)
            raise
        return remote_session(
            client,
            env,
            close_env=close_env,
            owns_env=owns_env,
            view=view,
            workflow_edition=workflow_edition,
            bridge=bridge,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(address={self._address!r})"


class _Secret:
    """A value whose repr never shows it (an API key)."""

    __slots__ = ("value",)

    def __init__(self, value: str | None) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "<redacted>" if self.value else "None"


def env_contract_of(env: object) -> EnvContract:
    """The env's published ``env_contract``, or a TypeError if it exposes none."""
    contract = getattr(env, "env_contract", None)
    if contract is None:
        raise TypeError(
            "a served model runs against an env client exposing `env_contract` "
            f"(RemoteEnv, SandboxEnv, or an address to dial); got {type(env).__name__}"
        )
    return cast("EnvContract", contract)


def dial_env_address(
    env: EnvTarget, workflow_edition: str | None = None
) -> tuple[EnvTarget, bool]:
    """Dial a bare ``env`` address into a ``RemoteEnv``; ``True`` when this dialed it.

    A handle dialed here is the session's own: it is released (its join slot
    freed) when the session closes, like a connection the session opens itself.
    ``workflow_edition`` is the session's declaration, which the env handshake
    carries ahead of the process and project ones.
    """
    if not isinstance(env, str):
        return env, False
    from .._native import RemoteEnv

    return RemoteEnv(env, workflow_edition=workflow_edition), True


def close_dialed_env(env: object) -> None:
    """Close an env a failed ``session()`` dialed; a close error is logged."""
    close = getattr(env, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # the original error wins
        logger.warning("closing the env a failed session dialed failed: %s", exc)


def env_session_offer(env: object) -> tuple[list[str], str | None] | None:
    """The env's handshake offer ``(CAN, WANT)``, when ``env`` is a dialed handle.

    A served-model session is the runtime tier of a three-way floor; handing the
    model client the env's real offer keeps a refusal honest about what the env
    actually declared. ``None`` when the handle exposes no handshake.
    """
    offer = getattr(env, "_session_offer", None)
    if not callable(offer):
        return None
    return cast("tuple[list[str], str | None]", offer())


def remote_session(
    client: PyModelClient,
    env: object,
    *,
    owner: Any = None,
    close_env: bool = False,
    owns_env: bool = False,
    view: object = None,
    workflow_edition: str | None = None,
    bridge: ValueBridge | None = None,
) -> Session[Any, Any]:
    """Build a neutral :class:`Session` over a pre-built served-model ``client``.

    Uses ``bridge`` when given (a local env's), else the env's own ``_bridge``
    (matching its value types) if it exposes one, else the identity bridge. ``owner`` is a managed source the session keeps alive and
    shuts down on close (e.g. a ``SandboxModel`` container); a cheap container-less
    handle passes none.
    """
    from .._models._eval import Session
    from .._value_conversion import identity_bridge

    if bridge is None:
        bridge = getattr(env, "_bridge", identity_bridge)
    return Session._create(  # pyright: ignore[reportPrivateUsage]
        env=env,
        model_client=client,
        bridge=bridge,
        owner=owner,
        close_env=close_env,
        owns_env=owns_env,
        view=view,
        workflow_edition=workflow_edition,
    )


__all__ = [
    "ActT",
    "ObsT",
    "RemoteModelBase",
    "env_contract_of",
    "env_session_offer",
    "remote_session",
]
