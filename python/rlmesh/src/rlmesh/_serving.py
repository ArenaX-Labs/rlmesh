"""Internal helpers for loading environments to serve through RLMesh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ._bootstrap.loaders import import_packages as _import_packages
from ._bootstrap.loaders import load_env_entrypoint as _load_env_entrypoint
from ._bootstrap.loaders import load_environment as _load_environment

if TYPE_CHECKING:
    from ._server import EnvLike, VectorServerEnvLike

    ServedEnv = EnvLike[Any, Any] | VectorServerEnvLike

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
) -> ServedEnv:
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
) -> ServedEnv:
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
