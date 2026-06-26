"""Model client: an env-agnostic served-policy handle.

The model-side mirror of :mod:`._remote_env`. A :class:`RemoteModelBase` is an
env-agnostic handle to a served model (policy); :meth:`RemoteModelBase.session`
binds it to one env -- sending that env's contract (and adapter tags) so the served
model resolves its adapter -- and returns the neutral :class:`rlmesh.Session` you
drive with ``reset`` / ``predict`` / ``step``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from .._framework_bridge import ValueBridge

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
        address: Model endpoint address such as ``"tcp://127.0.0.1:5555"``.
        token: Optional bearer token for an authenticated endpoint.
    """

    _bridge: ClassVar[ValueBridge]

    def __init__(self, address: str, *, token: str = "") -> None:
        self._bridge.ensure_available()
        self._address = address
        self._token = token

    @property
    def address(self) -> str:
        return self._address

    def session(
        self,
        env: object,
        *,
        instruction: str | None = None,
        close_env: bool = False,
        token: str | None = None,
        trust_entrypoints: bool | None = None,
    ) -> Session[Any, Any]:
        """Bind this served policy to ``env`` and return a :class:`rlmesh.Session`.

        ``env`` is an env client (e.g. ``RemoteEnv``/``SandboxEnv``) exposing an
        ``env_contract``; that contract -- including the env's adapter tags -- is sent
        to the model server, which resolves its adapter for this env. ``instruction``,
        ``token`` and ``trust_entrypoints`` apply to local models and are ignored here
        (the served model owns its adapter and auth).
        """
        _ = instruction, token, trust_entrypoints
        try:
            from rlmesh._rlmesh import PyModelClient
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        client = PyModelClient(self._address, env_contract_of(env), self._token)
        return remote_session(client, env, close_env=close_env)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(address={self._address!r})"


def env_contract_of(env: object) -> EnvContract:
    contract = getattr(env, "env_contract", None)
    if contract is None:
        raise TypeError(
            "rlmesh.session(model, env) requires an env client exposing `env_contract` "
            f"(e.g. RemoteEnv or SandboxEnv); got {type(env).__name__}"
        )
    return cast("EnvContract", contract)


def remote_session(
    client: PyModelClient, env: object, *, owner: Any = None, close_env: bool = False
) -> Session[Any, Any]:
    """Build a neutral :class:`Session` over a pre-built served-model ``client``.

    Selects the env's own ``_bridge`` (matching its value types) if it exposes one,
    else the identity bridge. ``owner`` is a managed source the session keeps alive and
    shuts down on close (e.g. a ``SandboxModel`` container); a cheap container-less
    handle passes none.
    """
    from .._framework_bridge import identity_bridge
    from .._models._eval import Session

    bridge = getattr(env, "_bridge", identity_bridge)
    return Session(
        env=env, model_client=client, bridge=bridge, owner=owner, close_env=close_env
    )


__all__ = [
    "ActT",
    "ObsT",
    "RemoteModelBase",
    "env_contract_of",
    "remote_session",
]
