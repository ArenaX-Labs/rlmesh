"""Public helpers for loading environments to serve through RLMesh.

Use :func:`load_env` (or :func:`load_env_entrypoint`) to construct an
environment by Gymnasium id or ``module:callable`` entrypoint, then hand it to
:class:`rlmesh.EnvServer` to serve it.

Examples:
    >>> import rlmesh
    >>> env = rlmesh.serving.load_env("CartPole-v1")  # doctest: +SKIP
    >>> server = rlmesh.EnvServer(env)  # doctest: +SKIP
    >>> server.serve()  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from ._bootstrap.env import import_packages as _import_packages
from ._bootstrap.env import load_env_entrypoint as _load_env_entrypoint
from ._bootstrap.env import load_environment as _load_environment

if TYPE_CHECKING:
    from .server import EnvLike

__all__ = [
    "import_packages",
    "load_env",
    "load_env_entrypoint",
]


def load_env(
    env_id: str,
    *,
    packages: Sequence[str] = (),
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> EnvLike:
    """Load a Gymnasium/Gym environment by registered id (e.g. ``"CartPole-v1"``).

    ``packages`` are imported first so their environments self-register; ``num_envs``
    > 1 vectorizes (``vectorization_mode`` ``"sync"``/``"async"``). Returns an
    environment suitable for :class:`rlmesh.EnvServer`.
    """
    return _load_environment(
        env_id,
        list(packages),
        num_envs,
        vectorization_mode,
        dict(kwargs) if kwargs is not None else None,
    )


def load_env_entrypoint(
    entrypoint: str,
    *,
    packages: Sequence[str] = (),
    kwargs: Mapping[str, object] | None = None,
) -> EnvLike:
    """Load an environment from a ``module:callable`` factory entrypoint.

    The callable must return an env exposing ``reset(...)``/``step(...)``;
    ``packages`` are imported before resolving it. Returns an environment suitable
    for :class:`rlmesh.EnvServer`.
    """
    return _load_env_entrypoint(
        entrypoint,
        list(packages),
        dict(kwargs) if kwargs is not None else None,
    )


def import_packages(packages: Sequence[str]) -> None:
    """Import packages so they can register their environments."""
    _import_packages(packages)
