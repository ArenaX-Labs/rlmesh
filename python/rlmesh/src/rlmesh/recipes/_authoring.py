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

from typing import TYPE_CHECKING, ClassVar, TypeGuard

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

    def prepare(self) -> None:
        """Optional construct-time CODE hook, run in the container before ``make()``.

        Use it for construction-time work that is code, not data -- downloading a
        checkpoint, warming a cache. The default is a no-op.
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
        instance = cls()
        instance.prepare()
        return instance.make(**kwargs)

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
    instance = cls()
    instance.prepare()
    env = instance.make(**kwargs)
    if not looks_like_env(env):
        # Match the gate the container/entrypoint path enforces, so both paths fail
        # identically at construction rather than later.
        raise TypeError(
            f"{cls.__qualname__}.make(...) did not return an environment with "
            "reset(...) and step(...)"
        )
    return env
