"""Class-style authoring: subclass ``EnvRecipe`` to define an environment.

``EnvRecipe`` co-locates the recipe DATA (the ``name``/``build``/``setup`` class
attributes) with the construction CODE (the ``make()`` factory and an optional
``prepare()`` hook) in one class -- and *projects* to an inert :class:`Recipe`.

The projection (:meth:`EnvRecipe.to_recipe`) is **execution-free and import-safe**:
it reads the class attributes and computes the entrypoint *string*; it never
instantiates the class and never imports the environment's dependencies. That is
what lets you author (and register) an IsaacSim recipe on a machine where
``import isaaclab`` cannot even succeed -- *authoring is not running*. Put every
heavy import inside ``make()``/``prepare()``.

Because a method's return annotation is evaluated at class-definition time, a
subclass that annotates ``def make(self) -> isaaclab.Env`` would fail to import on
that machine. **Every module defining an ``EnvRecipe`` subclass must start with**
``from __future__ import annotations``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard

from ._schema import Build, PyMake, Recipe, RecipeValidationError, Setup

if TYPE_CHECKING:
    from rlmesh.server import EnvLike

__all__ = ["EnvRecipe", "as_authored_recipe", "construct_authored", "is_env_recipe"]

#: The classmethod the projected entrypoint resolves to; runs the lifecycle.
_CONSTRUCT = "_rlmesh_construct"


class EnvRecipe:
    """Base class for authoring an environment and its recipe together.

    Subclasses set the data attributes and define the factory::

        from __future__ import annotations
        import rlmesh
        from rlmesh.recipes import Build, PipInstall


        class PointGoal(rlmesh.EnvRecipe):
            name = "safety/point-goal"
            build = Build(pip=[PipInstall(["safety-gymnasium==1.0.0"])])

            def make(self, env_id="SafetyPointGoal1-v0", **kwargs):
                import safety_gymnasium
                from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium

                return SafetyGymnasium2Gymnasium(
                    safety_gymnasium.make(env_id, **kwargs)
                )


        rlmesh.register(PointGoal)

    The class is referenced by name from the container; only ``make()``/
    ``prepare()`` run there, and only the inert projected recipe travels.
    """

    #: ``namespace/name`` identity. Required to register/serve/sandbox.
    name: ClassVar[str]
    #: The build phase (image). Defaults to today's base + rlmesh + gymnasium.
    build: ClassVar[Build] = Build()
    #: Construct-time DATA (env vars, files), applied before ``make()``.
    setup: ClassVar[Setup] = Setup()
    #: The published env contract -- the env-side mirror of ``ModelRecipe.spec``.
    #: An ``EnvTags`` the framework attaches to the env ``make()`` returns, so the
    #: served env publishes them and the recipe declares its contract statically
    #: (projected onto ``recipe.adapter``). Leave ``None`` and call
    #: ``rlmesh.adapters.tag(env, ...)`` inside ``make()`` for tags that depend on
    #: how the env was constructed.
    tags: ClassVar[Any] = None

    def prepare(self) -> None:
        """Optional construct-time CODE hook, run once before ``make()`` on this instance.

        Use it for side effects whose result lives *durably* somewhere other than this
        instance: a file on disk (a downloaded checkpoint), a warmed cache, or a
        process-global singleton (e.g. an Isaac ``SimulationApp``). Share state with
        ``make()`` through instance attributes -- ``self._x`` set here is read in
        ``make()`` (same instance, same synchronous construction). The default is a no-op.

        The recipe instance is discarded the moment ``make()`` returns. Only what the
        returned env references (or a process-global) survives into ``reset``/``step``/
        ``close``. Do NOT leave the sole reference to a per-env resource (open file,
        subprocess, render/GPU context, socket, license) on ``self`` -- create it in
        ``make()`` and let the returned env own it and release it in ``env.close()``.
        There is no recipe teardown hook. ``prepare()`` runs once per *construction*, not
        per process, so guard a process-global launch against double-construction.
        """

    def make(self, **kwargs: object) -> EnvLike:
        """Construct and return the environment. Subclasses must override.

        Put heavy imports *inside* this method so the class stays importable where
        the dependencies are absent.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must define make(self, **kwargs) -> env"
        )

    @classmethod
    def _rlmesh_construct(cls, **kwargs: object) -> EnvLike:
        """The lifecycle the projected recipe entrypoint runs: prepare then make."""
        instance = _instantiate(cls)
        instance.prepare()
        return _apply_class_tags(cls, instance.make(**kwargs))

    @classmethod
    def to_recipe(cls, **make_kwargs: object) -> Recipe:
        """Project this class to an inert :class:`Recipe` (executes nothing).

        Reads ``name``/``build``/``setup`` and computes the ``module:Class`` entry
        string. Raises if the class is not importable by that path (defined in
        ``__main__`` or a local scope), because the container must import it.
        """
        # Require the name on the concrete class, not inherited -- otherwise a
        # no-name subclass would silently project under its parent's identity.
        name = cls.__dict__.get("name")
        if not isinstance(name, str) or not name:
            inherited = getattr(cls, "name", None)
            hint = (
                f" (it would otherwise inherit {inherited!r})"
                if isinstance(inherited, str) and inherited
                else ""
            )
            raise RecipeValidationError(
                f'{cls.__qualname__} must declare its own `name = "namespace/name"`{hint}'
            )
        module = cls.__module__
        qualname = cls.__qualname__
        if module == "__main__" or "<locals>" in qualname:
            raise RecipeValidationError(
                f"EnvRecipe {name!r} is defined in {module}:{qualname}, which the "
                "container cannot import; define it in an installed module"
            )
        entrypoint = f"{module}:{qualname}.{_CONSTRUCT}"
        return Recipe(
            name=name,
            make=PyMake(entrypoint=entrypoint, kwargs=make_kwargs),
            build=cls.build,
            setup=cls.setup,
            adapter=cls.tags,
        )

    @classmethod
    def check(cls) -> None:
        """Validate this recipe without importing its dependencies (see :func:`check`)."""
        from ._check import check as _check

        _check(cls.to_recipe())


def is_env_recipe(source: object) -> TypeGuard[type[EnvRecipe]]:
    """Whether ``source`` is an ``EnvRecipe`` subclass (the class, not an instance)."""
    return isinstance(source, type) and issubclass(source, EnvRecipe)


def as_authored_recipe(source: object) -> Recipe | None:
    """Project an authored source to a :class:`Recipe`, or None if it is neither.

    Returns the projected recipe for an ``EnvRecipe`` subclass, the recipe itself
    for a :class:`Recipe`, and ``None`` for a plain string id/name (the caller
    resolves those). Used by every coercion point so there is one rule.
    """
    if is_env_recipe(source):
        return source.to_recipe()
    if isinstance(source, Recipe):
        return source
    return None


def _instantiate(cls: type[EnvRecipe]) -> EnvRecipe:
    """Construct ``cls()`` with a recipe-aware error if it requires constructor args.

    The lifecycle always instantiates with no arguments, so a required-arg ``__init__``
    fails with a confusing native ``TypeError``; per-construction parameters belong in
    ``make(self, **kwargs)``, not ``__init__``.
    """
    try:
        signature = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls()  # un-introspectable: let cls() raise on its own terms
    required = [
        name
        for name, p in signature.parameters.items()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    if required:
        raise TypeError(
            f"{cls.__qualname__} is instantiated with no arguments, but its __init__ "
            f"requires {required}. Put per-construction parameters in "
            "make(self, **kwargs) (baked into the recipe), not __init__."
        )
    return cls()


def construct_authored(cls: type[EnvRecipe], **kwargs: object) -> EnvLike:
    """Construct an ``EnvRecipe`` IN-PROCESS, without re-importing by entrypoint.

    Used by the local ``make``/``EnvServer`` paths so a class defined in a script
    or notebook still runs locally (the entrypoint round-trip is only needed to
    cross into a container). Applies ``setup``, then runs the same prepare + make
    lifecycle as :meth:`EnvRecipe._rlmesh_construct`.
    """
    from rlmesh._bootstrap.env import looks_like_env

    from ._build import apply_setup

    apply_setup(cls.setup)
    instance = _instantiate(cls)
    instance.prepare()
    env = instance.make(**kwargs)
    if not looks_like_env(env):
        # Match the gate the container/entrypoint path enforces, so both paths fail
        # identically at construction rather than later.
        raise TypeError(
            f"{cls.__qualname__}.make(...) did not return an environment with "
            "reset(...) and step(...)"
        )
    return _apply_class_tags(cls, env)


def _apply_class_tags(cls: type[EnvRecipe], env: EnvLike) -> EnvLike:
    """Publish the recipe's declared ``tags`` on the constructed env.

    The env-side mirror of how a resolved model carries its spec: a class-level
    ``tags`` is attached to ``make()``'s return so the served env's contract
    publishes it. A recipe that also tags inside ``make()`` is a double
    declaration, so fail loud rather than silently let one win.
    """
    tags = cls.tags
    if tags is None:
        return env
    from rlmesh.adapters import EnvTags, tag

    if EnvTags.from_metadata(getattr(env, "metadata", None) or {}) is not None:
        raise RecipeValidationError(
            f"{cls.__qualname__} declares class-level tags= and make() also tagged "
            "the env; declare the tags in one place, not both"
        )
    return tag(env, tags)
