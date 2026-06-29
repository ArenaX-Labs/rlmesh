"""Env connection and contract synthesis for the local drive path.

Turns a session target (a live env, an ``EnvFactory``, a remote handle, or an
address) into a ``(client, contract, owns_client)`` triple, synthesizes a
client-side contract for a local env, and provides the small env-protocol shims
(reset/close/shutdown) and the env-side value bridge. Used by
:class:`rlmesh._models._eval.Session`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .._value_conversion import ValueBridge

# The cross-module surface this private module offers its siblings (chiefly
# ``_eval.Session``); ``__all__`` marks these underscore helpers as exported so
# the type checker does not read sibling imports as private-symbol leakage.
__all__ = [
    "adapter_env_bridge",
    "close_client",
    "connect_env",
    "reset_env",
    "shutdown_env",
]


def connect_env(
    target: object, token: str, remote_env_cls: type | None
) -> tuple[Any, Any, bool]:
    """Resolve a session target to ``(client, contract, owns_client)``.

    ``target`` is an address string, a live env (local object or remote handle), an
    :class:`~rlmesh.EnvFactory`, or an object exposing an ``address``. ``owns_client``
    is True when this dialed the connection, so the session knows to close it.
    """
    if isinstance(target, str):
        client = _remote_env(target, remote_env_cls)
        return client, client.env_contract, True
    if hasattr(target, "reset") and hasattr(target, "step"):
        # A live env: a remote/served handle exposes a native env_contract; a local
        # env exposes its spaces + metadata directly, so synthesize the contract from
        # the env (tags ride in env.metadata via tag() / EnvFactory.make).
        native = _native_contract(target)
        contract = native if native is not None else _local_contract(target)
        return target, contract, False
    if hasattr(target, "make"):
        # An EnvFactory: prepare()+make() its env (which carries the factory's tags)
        # and drive it locally -- no serving needed to resolve a spec'd adapter.
        env = _factory_env(target)
        return env, _local_contract(env), False
    address = getattr(target, "address", None)
    if isinstance(address, str):
        client = _remote_env(address, remote_env_cls)
        return client, client.env_contract, True
    raise TypeError(
        "session()/run() expect an env object, an EnvFactory, a remote-env object, "
        f"or an address string; got {type(target).__name__}"
    )


@dataclass(frozen=True)
class _LocalEnvContract:
    """Client-side stand-in for an env contract when driving a *local* env.

    A served env publishes a native ``EnvContract`` (spaces + metadata) over the
    handshake; a local env object exposes the same pieces directly. Bundling them
    here lets the adapter-resolution path be identical for local and remote envs --
    the env's tags ride in ``env.metadata`` (attached by :func:`rlmesh.adapters.tag`
    or :meth:`rlmesh.EnvFactory.make`).
    """

    metadata: Mapping[str, Any] | None
    observation_space: object
    action_space: object
    num_envs: int


def _native_contract(env: object) -> object | None:
    """Return a real ``env_contract`` (a remote/served handle) or ``None`` for a local env.

    Reads the attribute off the type or instance ``__dict__`` rather than via
    ``getattr``, so a gymnasium env does not trigger its deprecated
    wrapper-attribute forwarding warning just because we probed for a contract.
    """
    if getattr(type(env), "env_contract", None) is not None or (
        "env_contract" in getattr(env, "__dict__", {})
    ):
        try:
            return cast("Any", env).env_contract
        except AttributeError:
            return None
    return None


def _local_contract(env: object) -> Any:
    return _LocalEnvContract(
        metadata=getattr(env, "metadata", None),
        observation_space=getattr(env, "observation_space", None),
        action_space=getattr(env, "action_space", None),
        num_envs=_num_envs(env),
    )


def _num_envs(env: object) -> int:
    # A single env has no ``num_envs``; only a vector env does. Probe the type /
    # instance __dict__ (not a plain getattr) so a gymnasium env does not emit its
    # deprecated wrapper-attribute forwarding warning, same as _native_contract.
    if getattr(type(env), "num_envs", None) is not None or (
        "num_envs" in getattr(env, "__dict__", {})
    ):
        try:
            return int(cast("Any", env).num_envs or 1)
        except (AttributeError, TypeError, ValueError):
            return 1
    return 1


def _factory_env(factory: object) -> Any:
    """Build a local env from an EnvFactory: ``prepare()`` + ``make()``.

    ``EnvFactory.make`` stamps the factory's ``tags`` onto the env it returns, so a
    spec'd model can resolve its adapter from the local env alone -- no serving.
    """
    from .._bootstrap.loaders import construct_authored_env

    return construct_authored_env(factory)


def _remote_env(address: str, remote_env_cls: type | None) -> Any:
    if remote_env_cls is None:
        from ..numpy import RemoteEnv

        remote_env_cls = RemoteEnv
    return remote_env_cls(address)


def adapter_env_bridge(client: Any) -> ValueBridge:
    """The bridge for the framework the env hands its observations *in*.

    The env-side encoder/decoder must match the env's own value type, never the
    model's. A remote/served handle decodes the wire payload into its framework
    before returning it (a torch ``RemoteEnv`` hands the loop torch tensors), so
    its ``_bridge`` is the right re-encoder for the native plan -- and is why the
    served cross-framework path works. A raw local env returns observations in its
    native array type, which for a gym/gymnasium env is numpy, so default to the
    numpy bridge: the model's framework bridge would reject numpy (the
    cross-framework local-driving bug). A custom local env that emits another
    framework's tensors can expose ``_bridge`` on its class to override.
    """
    bridge = getattr(client, "_bridge", None)
    if bridge is not None:
        return cast("ValueBridge", bridge)
    from ..numpy import _numpy_bridge  # pyright: ignore[reportPrivateUsage]

    return _numpy_bridge


def reset_env(client: Any, seed: int | None) -> tuple[Any, Mapping[str, Any]]:
    """Reset an env and normalize its return to ``(obs, info)``.

    Accepts a gymnasium ``(obs, info)`` pair (the second element a Mapping) or a bare
    observation; the latter pairs with an empty info dict.
    """
    result: Any = client.reset(seed=seed) if seed is not None else client.reset()
    if isinstance(result, tuple):
        pair = cast("tuple[Any, ...]", result)
        # Only a (obs, info) pair where the second element is a Mapping is a
        # gymnasium reset return; any other tuple is itself the observation.
        if len(pair) == 2 and isinstance(pair[1], Mapping):
            return pair[0], cast("Mapping[str, Any]", pair[1])
        return pair, {}
    return result, {}


def close_client(client: Any) -> None:
    """Release a dialed connection by calling its ``close``, if it has one."""
    close = getattr(client, "close", None)
    if callable(close):
        close()


def shutdown_env(target: object) -> None:
    """Stop an env: a sandbox container via ``close``, else a remote owner.

    Passes a teardown reason to whichever of ``shutdown``/``close`` accepts one,
    decided by binding the signature rather than catching ``TypeError`` -- so a
    ``TypeError`` raised inside the callable is never swallowed.
    """
    # A sandbox session's close() stops its container; its inherited shutdown() is the
    # remote owner-shutdown, so close() is the right teardown for an owned sandbox.
    from .._sandbox.session import SandboxLifecycle

    if isinstance(target, SandboxLifecycle):
        target.close()
        return
    import inspect

    for name in ("shutdown", "close"):
        fn = getattr(target, name, None)
        if not callable(fn):
            continue
        try:
            inspect.signature(fn).bind("model run complete")
            accepts_reason = True
        except (TypeError, ValueError, KeyError):  # also un-introspectable builtins
            accepts_reason = False
        fn("model run complete") if accepts_reason else fn()
        return
