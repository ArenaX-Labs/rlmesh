"""0.1 call shapes 0.2 removed: each raises an error naming its replacement."""

from __future__ import annotations

from typing import NoReturn


def removed(old: str, new: str) -> NoReturn:
    """Refuse the 0.1 call ``old``, naming ``new``."""
    raise TypeError(f"{old} is gone in rlmesh 0.2: {new}")


def RemoteModel(*args: object, **kwargs: object) -> NoReturn:  # noqa: N802 -- the 0.1 name
    """Removed in 0.2: ``Model(address)`` wraps a model served elsewhere."""
    _ = args, kwargs
    removed(
        "RemoteModel(...)",
        'Model("host:port/name") wraps the model served there (Model.serve(address, '
        "name=...))",
    )


__all__ = ["RemoteModel", "removed"]
