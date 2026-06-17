"""Model client: an env-agnostic policy handle and its per-env bound session.

The model-side mirror of :mod:`._remote_env`. A :class:`RemoteModelBase` is an
env-agnostic handle to a model (policy) server; :meth:`RemoteModelBase.against`
binds it to one env -- sending that env's contract (and adapter tags) so the
served model resolves its adapter -- and returns a :class:`ModelSession` you
drive with ``reset`` / ``predict`` symmetrically with the env's
``reset`` / ``step``.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar, cast

from .._framework_bridge import ValueBridge

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from ..specs import EnvContract

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class _OwnedSource(Protocol):
    """A model source a session may own and stop on close (e.g. SandboxModel)."""

    def shutdown(self) -> None: ...


class ModelSession(Generic[ObsT, ActT]):
    """A model bound to one env: the session :meth:`RemoteModelBase.against` returns.

    Drive it by hand with :meth:`reset` / :meth:`predict`, mirroring the env's
    ``reset`` / ``step``. One session owns one route (one env contract); make a
    fresh session per env rather than reusing one across envs.
    """

    def __init__(
        self,
        client: PyModelClient,
        bridge: ValueBridge,
        owner: _OwnedSource | None = None,
    ) -> None:
        self._client = client
        self._bridge = bridge
        # When a managed source (e.g. SandboxModel) builds the client itself, the
        # session holds the only strong ref to it; without this the source would
        # be GC'd -- and its container stopped via __del__ -- before first predict.
        self._owner = owner

    @property
    def address(self) -> str:
        return self._client.address()

    def reset(self) -> None:
        """Begin a new episode; the next :meth:`predict` marks a reset boundary."""
        self._client.reset()

    def predict(self, observation: ObsT) -> ActT:
        """Ask the policy for an action given one observation."""
        action = self._client.predict(self._bridge.encode(observation))
        return cast(ActT, self._bridge.decode(action))

    def close(self) -> None:
        """Close this session's route, and stop the model server if we own it.

        For an ownerless session (the cheap ``RemoteModel`` handle) this only
        closes the route. When the session owns its source (the one-liner
        ``SandboxModel(...).against(env)`` started a container), closing also
        shuts that source down so the container the session started is stopped.
        """
        self._client.close()
        owner = self._owner
        if owner is not None:
            self._owner = None
            owner.shutdown()

    def __enter__(self) -> ModelSession[ObsT, ActT]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = exc_type, exc_val, exc_tb
        self.close()

    def __repr__(self) -> str:
        return f"ModelSession(address={self.address!r})"


class RemoteModelBase(Generic[ObsT, ActT]):
    """Env-agnostic handle to a model (policy) server.

    Bind it to an env with :meth:`against` to get a :class:`ModelSession`. The
    handle carries no env contract, so one handle drives many envs:
    ``model.against(env_a)`` and ``model.against(env_b)`` are independent
    sessions off the same policy.

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

    def against(self, env: object) -> ModelSession[ObsT, ActT]:
        """Bind this policy to ``env``, opening a route configured from its contract.

        ``env`` is an env client (e.g. ``RemoteEnv`` or ``SandboxEnv``) exposing
        an ``env_contract``. That contract -- including the env's adapter tags --
        is sent to the model server, which resolves its adapter for this env.

        Returns:
            A :class:`ModelSession` to drive in a loop alongside ``env``.
        """
        try:
            from rlmesh._rlmesh import PyModelClient
        except ImportError as e:  # pragma: no cover - import guard
            raise ImportError("Failed to import _rlmesh native module.") from e

        client = PyModelClient(self._address, env_contract_of(env), self._token)
        return ModelSession(client, self._bridge)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(address={self._address!r})"


def env_contract_of(env: object) -> EnvContract:
    contract = getattr(env, "env_contract", None)
    if contract is None:
        raise TypeError(
            "against(env) requires an env client exposing `env_contract` "
            f"(e.g. RemoteEnv or SandboxEnv); got {type(env).__name__}"
        )
    return cast("EnvContract", contract)


def session_for_client(
    client: PyModelClient, env: object, owner: _OwnedSource | None = None
) -> ModelSession[object, object]:
    """Wrap a pre-built model ``client`` in a session bridged for ``env``.

    The shared tail of :meth:`RemoteModelBase.against` for callers (e.g.
    ``SandboxModel``) that build the client themselves: selects the env's own
    ``_bridge`` if it exposes one, else the identity bridge. ``owner`` is the
    source the session must keep alive (and shut down on close); the cheap
    container-less handle passes none.
    """
    from .._framework_bridge import identity_bridge

    bridge = getattr(env, "_bridge", identity_bridge)
    return ModelSession(client, bridge, owner)


__all__ = [
    "ActT",
    "ModelSession",
    "ObsT",
    "RemoteModelBase",
    "env_contract_of",
    "session_for_client",
]
