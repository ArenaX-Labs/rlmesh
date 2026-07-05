"""Model client: an env-agnostic served-policy handle.

The model-side mirror of :mod:`._remote_env`. A :class:`RemoteModelBase` is an
env-agnostic handle to a served model (policy); :meth:`RemoteModelBase.session`
binds it to one env -- sending that env's contract (and adapter tags) so the served
model resolves its adapter -- and returns the neutral :class:`rlmesh.Session` you
drive with ``reset`` / ``predict`` / ``step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from .._value_conversion import ValueBridge
from ..types import EnvTarget, ViewArg
from ._endpoint import Transport, normalize_connect_address

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._models._eval import Session
    from ..specs import EnvContract

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class RemoteModelBase(Generic[ObsT, ActT]):
    """Env-agnostic handle to a served model (policy).

    Bind it to an env with :meth:`session` to get a :class:`rlmesh.Session`. The handle
    carries no env contract, so one handle drives many envs: ``model.session(env_a)``
    and ``model.session(env_b)`` are independent sessions off the same policy.

    Args:
        address: Model endpoint address such as ``"tcp://127.0.0.1:5555"``,
            ``"127.0.0.1:5555"``, or ``"unix:///tmp/model.sock"``.
        host: TCP host helper used when ``address`` is omitted.
        port: TCP port helper used when ``address`` is omitted.
        path: Unix socket path helper used when ``address`` is omitted.
        transport: Explicit transport selector.
        connect_timeout_seconds: Optional bound on each session's dial to the
            model server; ``None`` (default) waits indefinitely.
        request_timeout_seconds: Optional per-request bound applied to every
            predict/close RPC of sessions off this handle; ``None`` (default)
            waits indefinitely.
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
    ) -> None:
        self._bridge.ensure_available()
        self._address = normalize_connect_address(
            address,
            host=host,
            port=port,
            path=path,
            transport=transport,
        )
        self._connect_timeout_seconds = connect_timeout_seconds
        self._request_timeout_seconds = request_timeout_seconds

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
    ) -> Session[Any, Any]:
        """Bind this served policy to ``env`` and return a :class:`rlmesh.Session`.

        ``env`` is an env client (e.g. ``RemoteEnv``/``SandboxEnv``) exposing an
        ``env_contract``; that contract -- including the env's adapter tags -- is sent
        to the model server, which resolves its adapter for this env. ``instruction``
        and ``trust_entrypoints`` apply to local models and are rejected here (the
        served model owns its adapter). ``view`` opts into the built-in live viewer
        over this client's own env loop (see :func:`rlmesh.run`).

        ``execution_horizon`` (> 1) opts a chunk-capable served model into action
        chunking: the model emits its native chunk, and this client replays a prefix
        of it one action per step open-loop (skipping the RPC), re-planning every
        ``execution_horizon`` steps. The model must define ``predict_chunk``;
        otherwise it re-plans every step.
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
        from .._load_native import load_native

        try:
            client = load_native("PyModelClient")(
                self._address,
                env_contract_of(env),
                execution_horizon,
                connect_timeout_seconds=self._connect_timeout_seconds,
                request_timeout_seconds=self._request_timeout_seconds,
            )
        except ConnectionError as exc:
            raise ConnectionError(
                f"could not connect to a model at {self._address!r}; "
                f"is the model being served? ({exc})"
            ) from None
        return remote_session(client, env, close_env=close_env, view=view)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(address={self._address!r})"


def env_contract_of(env: object) -> EnvContract:
    """The env's published ``env_contract``, or a TypeError if it exposes none."""
    contract = getattr(env, "env_contract", None)
    if contract is None:
        raise TypeError(
            "rlmesh.session(model, env) requires an env client exposing `env_contract` "
            f"(e.g. RemoteEnv or SandboxEnv); got {type(env).__name__}"
        )
    return cast("EnvContract", contract)


def remote_session(
    client: PyModelClient,
    env: object,
    *,
    owner: Any = None,
    close_env: bool = False,
    view: object = None,
) -> Session[Any, Any]:
    """Build a neutral :class:`Session` over a pre-built served-model ``client``.

    Selects the env's own ``_bridge`` (matching its value types) if it exposes one,
    else the identity bridge. ``owner`` is a managed source the session keeps alive and
    shuts down on close (e.g. a ``SandboxModel`` container); a cheap container-less
    handle passes none.
    """
    from .._models._eval import Session
    from .._value_conversion import identity_bridge

    bridge = getattr(env, "_bridge", identity_bridge)
    return Session._create(  # pyright: ignore[reportPrivateUsage]
        env=env,
        model_client=client,
        bridge=bridge,
        owner=owner,
        close_env=close_env,
        view=view,
    )


__all__ = [
    "ActT",
    "ObsT",
    "RemoteModelBase",
    "env_contract_of",
    "remote_session",
]
