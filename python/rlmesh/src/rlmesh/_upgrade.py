"""0.1 call shapes 0.2 removed: each raises an error naming its replacement."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NoReturn

#: Set while a part still on the v1 wire (the ``rlmesh.serve`` entrypoint and
#: the sandboxes it serves in) serves or dials a v1 endpoint.
# TODO: delete with the v1 env wire once `rlmesh.serve` and `Sandbox` join
# sessions.
_V1_ENDPOINTS: ContextVar[bool] = ContextVar("rlmesh_v1_endpoints", default=False)


@contextmanager
def v1_endpoints() -> Iterator[None]:
    """Let the parts still on the v1 wire serve and dial v1 endpoints."""
    token = _V1_ENDPOINTS.set(True)
    try:
        yield
    finally:
        _V1_ENDPOINTS.reset(token)


def refuse_v1_address(where: str, address: str) -> None:
    """Refuse ``address`` when it is a v1 endpoint (0.2 uses session addresses)."""
    from ._rlmesh import is_v2_address

    if is_v2_address(address) or _V1_ENDPOINTS.get():
        return
    raise ValueError(
        f"{where}({address!r}): {address!r} is a v1 endpoint, which rlmesh 0.2 no "
        "longer serves or dials; use a session address such as ':0/name' (an "
        "OS-assigned local port), 'host:port/name' or 'session/name'"
    )


def refuse_remote_close_env(target: object) -> None:
    """Refuse ``close_env=True`` on an env served elsewhere on a session."""
    from ._rlmesh import is_v2_address

    address = target if isinstance(target, str) else getattr(target, "address", None)
    if (
        isinstance(address, str)
        and is_v2_address(address)
        and (isinstance(target, str) or hasattr(target, "env_contract"))
    ):
        raise ValueError(
            "close_env=True on a session address is gone in rlmesh 0.2: the env "
            "belongs to its serving process; an owner shutdown is "
            "server.shutdown() there"
        )


def removed(old: str, new: str) -> NoReturn:
    """Refuse the 0.1 call ``old``, naming ``new``."""
    raise TypeError(f"{old} is gone in rlmesh 0.2: {new}")


def Session(*args: object, **kwargs: object) -> NoReturn:  # noqa: N802 -- the 0.1 name
    """Removed in 0.2: ``model.bind(target)`` returns a :class:`Binding`."""
    _ = args, kwargs
    removed(
        "Session",
        "model.bind(target) returns a Binding you step by hand "
        "(reset(), step(), step(action))",
    )


def session(*args: object, **kwargs: object) -> NoReturn:
    """Removed in 0.2: ``model.bind(target)`` drives a model on one env by hand."""
    _ = args, kwargs
    removed(
        "rlmesh.session(model, env)",
        "model.bind(target) returns a Binding you step by hand; "
        "rlmesh.session(address) is reserved for the session as a place",
    )


def model_session(self: object, *args: object, **kwargs: object) -> NoReturn:
    """Removed in 0.2: ``model.bind(target)`` drives a model on one env by hand."""
    _ = self, args, kwargs
    removed(
        "Model.session(env)",
        "model.bind(target) returns a Binding you step by hand "
        "(reset(), step(), step(action))",
    )


def RemoteModel(*args: object, **kwargs: object) -> NoReturn:  # noqa: N802 -- the 0.1 name
    """Removed in 0.2: ``Model(address)`` wraps a model served elsewhere."""
    _ = args, kwargs
    removed(
        "RemoteModel(...)",
        'Model("host:port/name") wraps the model served there (Model.serve(address, '
        "name=...))",
    )


__all__ = [
    "RemoteModel",
    "Session",
    "model_session",
    "refuse_remote_close_env",
    "refuse_v1_address",
    "removed",
    "session",
    "v1_endpoints",
]
