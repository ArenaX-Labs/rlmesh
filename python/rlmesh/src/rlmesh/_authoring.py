"""Authoring base for environments: a thin runtime class (tags/make), NOT a build DSL.

Subclass :class:`EnvFactory` to describe an environment's *runtime*. There is no build
DSL here -- packaging stays in your Dockerfile. Models are authored by subclassing
``rlmesh.Model`` and overriding ``predict`` (no separate recipe noun); envs serve via
:class:`EnvServer`, ``EnvFactory.serve``, or ``python -m rlmesh.serve --env my_pkg:MyEnv``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, final

from rlmesh.types import EnvLike

if TYPE_CHECKING:
    from .adapters import EnvTags


class EnvFactory(ABC):
    """Authoring base that *builds* environment(s): set ``tags`` and implement ``make``.

    Subclass per obs/action contract -- the ``tags`` (a :class:`~rlmesh.adapters.EnvTags`)
    are that contract. ``make(**kwargs)`` is the factory and may return a single env or a
    vectorized batch; task selection and ``num_envs`` are its parameters, not separate
    subclasses. ``tags = None`` (the default) means a generic, un-adapted env.
    """

    tags: ClassVar[EnvTags | None] = None

    def prepare(self) -> None:  # noqa: B027  optional no-op hook, not abstract
        """Optional: one-time setup before ``make()``."""

    @abstractmethod
    def make(self, **kwargs: Any) -> EnvLike[Any, Any]:
        """Construct and return the environment to serve."""
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027  optional no-op hook, not abstract
        """Optional: release resources."""

    @final
    def serve(self, address: str, **kwargs: Any) -> None:
        """Host this env on ``address`` (blocking): ``prepare()`` + ``make(**kwargs)``, publish ``tags``."""
        from .serve import serve_env

        serve_env(self, address, **kwargs)
