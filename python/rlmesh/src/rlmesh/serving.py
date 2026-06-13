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
    """Load a Gymnasium/Gym environment by registered id.

    Args:
        env_id: Registered environment id, for example ``"CartPole-v1"``.
        packages: Packages imported before loading so their environments
            register themselves.
        num_envs: Number of vectorized environments to construct. ``1`` returns
            a single environment.
        vectorization_mode: Preferred vectorization mode (``"sync"`` or
            ``"async"``) when ``num_envs`` is greater than one.
        kwargs: Extra keyword arguments forwarded to environment creation.

    Returns:
        An RLMesh-servable environment suitable for :class:`rlmesh.EnvServer`.
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

    Args:
        entrypoint: Factory in ``module:callable`` form. The callable must
            return an environment exposing ``reset(...)`` and ``step(...)``.
        packages: Packages imported before resolving the entrypoint.
        kwargs: Extra keyword arguments forwarded to the factory.

    Returns:
        An RLMesh-servable environment suitable for :class:`rlmesh.EnvServer`.
    """
    return _load_env_entrypoint(
        entrypoint,
        list(packages),
        dict(kwargs) if kwargs is not None else None,
    )


def import_packages(packages: Sequence[str]) -> None:
    """Import packages so they can register their environments."""
    _import_packages(packages)
