"""Lower the flat ``packages=``/``imports=``/``factory=`` sugar to a real recipe.

Single-sourced so ``rlmesh.make`` (gym fallthrough) and ``rlmesh.register`` (flat
naming form) build the *same* validated dataclasses -- every ``__post_init__``
check (JSON-only kwargs, token allowlists, the PyMake-forbids-imports rule) runs
identically, and there is one place that knows how the sugar maps to a recipe.
The lowering is string-shape-only: it never imports the factory module, so the
sugar works on a machine that cannot import the environment's dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._schema import Build, GymMake, PipInstall, PyMake, Recipe, Requires

__all__ = ["factory_sugar_to_recipe", "gym_sugar_to_recipe"]

# Operators that mark a string as a PEP 508 package spec rather than a module name.
_SPEC_OPERATORS = ("==", ">=", "<=", "~=", "!=", "<", ">", "[")


def _reject_pip_shaped_imports(imports: Sequence[str], packages: Sequence[str]) -> None:
    """Catch the common mistake of passing a package spec to ``imports=``."""
    package_set = set(packages)
    for name in imports:
        if any(op in name for op in _SPEC_OPERATORS) or name in package_set:
            raise TypeError(
                f"imports= are module names imported for registration side effects "
                f"(e.g. 'ale_py'); {name!r} looks like a package — installs go in packages="
            )


def _build_from_packages(packages: Sequence[str]) -> Build:
    return Build(pip=[PipInstall(packages=list(packages))]) if packages else Build()


def gym_sugar_to_recipe(
    name: str,
    env_id: str,
    *,
    packages: Sequence[str] | None = None,
    imports: Sequence[str] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> Recipe:
    """Lower a gym id + flat ``packages=``/``imports=`` to a ``GymMake`` recipe."""
    packages = list(packages or ())
    imports = list(imports or ())
    _reject_pip_shaped_imports(imports, packages)
    return Recipe(
        name=name,
        make=GymMake(env_id=env_id, kwargs=dict(kwargs or {})),
        build=_build_from_packages(packages),
        requires=Requires(imports=imports) if imports else Requires(),
    )


def factory_sugar_to_recipe(
    name: str,
    entrypoint: str,
    *,
    packages: Sequence[str] | None = None,
    imports: Sequence[str] | None = None,
    kwargs: Mapping[str, object] | None = None,
) -> Recipe:
    """Lower a ``module:callable`` factory + flat ``packages=`` to a ``PyMake`` recipe."""
    if imports:
        raise TypeError(
            "imports= is gym-only; a py factory owns its own imports (drop imports=)"
        )
    packages = list(packages or ())
    return Recipe(
        name=name,
        make=PyMake(entrypoint=entrypoint, kwargs=dict(kwargs or {})),
        build=_build_from_packages(packages),
    )
