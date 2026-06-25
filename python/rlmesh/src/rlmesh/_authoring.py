"""Authoring bases: thin runtime classes (load/predict/spec), NOT build recipes.

Subclass to describe a policy or environment's *runtime*. There is no build DSL
here -- packaging stays in your Dockerfile. ``Model(MyPolicy)`` and
``python -m rlmesh.serve my_pkg:MyPolicy`` both accept a :class:`ModelRecipe`
subclass directly; envs serve via :class:`EnvServer` (or the same ``serve`` module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from .adapters import EnvTags, ModelSpec


class ModelRecipe:
    """Authoring base for a policy: set ``spec``, implement ``load`` and ``predict``.

    ``Model(MyPolicy).run(env)`` instantiates the class, calls ``load()`` once, then
    serves ``predict`` through the adapter resolved from ``spec`` and the env's tags.
    ``reset`` and ``close`` are optional episode-boundary and teardown hooks. This is
    runtime-only; it is not a build recipe.
    """

    spec: ClassVar[ModelSpec | None] = None

    def load(self) -> None:
        """Load weights into ``self`` (``from_pretrained`` etc.); heavy imports here."""

    def predict(self, observation: Any) -> Any:
        """Map one observation to an action (or an action chunk per ``spec``)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Optional: called at each episode boundary."""

    def close(self) -> None:
        """Optional: release resources at the end of a run."""


class EnvRecipe:
    """Authoring base for an environment: set ``tags``, implement ``make``.

    Served via ``EnvServer(make(), address, tags=tags)``. ``prepare`` is an optional
    one-time setup hook run before ``make``. Runtime-only; not a build recipe.
    """

    tags: ClassVar[EnvTags | None] = None

    def prepare(self) -> None:
        """Optional: one-time setup before ``make()``."""

    def make(self, **kwargs: Any) -> Any:
        """Construct and return the environment to serve."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional: release resources."""
