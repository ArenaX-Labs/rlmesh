"""Authoring base for environments: a thin runtime class (tags/make), NOT a build DSL.

Subclass :class:`EnvFactory` to describe an environment's *runtime*. There is no build
DSL here -- packaging stays in your Dockerfile. Models are authored by subclassing
``rlmesh.Model`` and overriding ``predict`` (no separate recipe noun); envs serve via
:class:`EnvServer`, ``EnvFactory.serve``, or ``python -m rlmesh.serve --env my_pkg:MyEnv``.
"""

from __future__ import annotations

import functools
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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Stamp the factory's ``tags`` onto every env ``make()`` returns, so the tag
        # "rides the environment": a locally-made env (no server) still carries the
        # contract a spec'd model resolves against. The subclass's own ``make`` body
        # is untouched; serving an already-stamped env merges the same tags
        # idempotently. ``tags = None`` is a no-op (a generic, un-adapted env).
        super().__init_subclass__(**kwargs)
        user_make = cls.__dict__.get("make")
        if user_make is None or getattr(user_make, "_rlmesh_tag_stamped", False):
            return

        @functools.wraps(user_make)
        def make(self: EnvFactory, *args: Any, **make_kwargs: Any) -> EnvLike[Any, Any]:
            env = user_make(self, *args, **make_kwargs)
            tags = type(self).tags
            if tags is not None:
                from .adapters import tag

                # validate=False: the obs/action layout is validated against the
                # tags at adapter-resolution time (serve or session), so a make()
                # of a vectorized batch (whose spaces differ) is not rejected here.
                env = tag(env, tags, validate=False)
            return env

        make._rlmesh_tag_stamped = True  # type: ignore[attr-defined]
        cls.make = make  # type: ignore[method-assign]

    def prepare(self) -> None:  # noqa: B027  optional no-op hook, not abstract
        """Optional: one-time setup before ``make()``."""

    @abstractmethod
    def make(self, **kwargs: Any) -> EnvLike[Any, Any]:
        """Construct and return the environment.

        Your override returns a plain env; the returned env is automatically
        stamped with this factory's ``tags`` (in ``env.metadata``), so the tag
        rides the environment -- a spec'd model can resolve its adapter from the
        env alone, whether it is served or driven locally via
        :func:`rlmesh.session`.
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027  optional no-op hook, not abstract
        """Optional: release resources."""

    @final
    def serve(self, address: str, **kwargs: Any) -> None:
        """Host this env on ``address`` (blocking): ``prepare()`` + ``make(**kwargs)``, publish ``tags``."""
        from .serve import serve_env

        serve_env(self, address, **kwargs)
