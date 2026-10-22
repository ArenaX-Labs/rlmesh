"""The served-model session helpers a ``SandboxModel`` binds through.

A sandboxed model is served on a v1 endpoint until sandboxes join sessions;
these helpers bind its client to one env and return the neutral
:class:`rlmesh.Session` driven with ``reset`` / ``predict`` / ``step``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from .._value_conversion import ValueBridge
from ..types import EnvTarget

if TYPE_CHECKING:
    from rlmesh._rlmesh import PyModelClient

    from .._models._eval import Session
    from ..specs import EnvContract

logger = logging.getLogger(__name__)


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
    "env_contract_of",
    "env_session_offer",
    "remote_session",
]
