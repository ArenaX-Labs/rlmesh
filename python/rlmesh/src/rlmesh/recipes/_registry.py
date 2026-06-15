"""A process-local name -> recipe registry.

The registry is the ``name -> recipe`` half of the ontology (an adapter is IO, a
recipe is construction, the registry is the lookup). It is deliberately small:
the resolver chain (entry points, git, OCI, cloud catalog) is a forward layer
that wraps this local map behind one ``Resolver`` protocol; slice 1 ships only
the local map.

Anti-shadowing: re-registering an already-registered name is rejected unless
``overwrite=True`` -- a recipe must not silently replace another author's recipe.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import overload

from ._authoring import EnvRecipe, is_env_recipe
from ._gym_sugar import factory_sugar_to_recipe, gym_sugar_to_recipe
from ._schema import Build, GymMake, PyMake, Recipe

__all__ = [
    "RecipeNotFoundError",
    "class_origin_dir",
    "clear_registry",
    "from_recipe_origin",
    "pprint_registry",
    "recipe_origin_dir",
    "register",
    "registered_names",
    "registry",
    "resolve",
    "resolve_from_recipe",
    "unregister",
]


class RecipeNotFoundError(LookupError):
    """Raised when a recipe name is not present in the local registry."""


_REGISTRY: dict[str, Recipe] = {}
# Best-effort ``name -> defining-package directory`` for recipes registered by an
# author other than the caller. Used to stage a ProjectInstall from the recipe's
# own source tree (not the launching process's cwd). Only populated when the
# origin can be determined; otherwise the name is simply absent.
_ORIGINS: dict[str, str] = {}


@overload
def register(source: type[EnvRecipe], *, overwrite: bool = ...) -> type[EnvRecipe]: ...
@overload
def register(source: Recipe, *, overwrite: bool = ...) -> Recipe: ...
@overload
def register(
    source: str,
    *,
    gym: str | None = ...,
    factory: str | None = ...,
    packages: Sequence[str] = ...,
    imports: Sequence[str] = ...,
    overwrite: bool = ...,
) -> Recipe: ...
def register(
    source: str | Recipe | type[EnvRecipe],
    *,
    gym: str | None = None,
    factory: str | None = None,
    packages: Sequence[str] = (),
    imports: Sequence[str] = (),
    overwrite: bool = False,
) -> Recipe | type[EnvRecipe]:
    """Register a recipe so it can be built by name.

    Three forms:

    * **Decorator / class** -- ``@register`` above an ``EnvRecipe`` subclass (or
      ``register(MyEnvRecipe)``): projects and registers it, and returns the class
      so the decorated name stays bound to it.
    * **Object** -- ``register(recipe)``: register a :class:`Recipe`, returned as-is.
    * **Flat naming sugar** -- ``register("namespace/name", gym=..., packages=[...])``
      or ``register("namespace/name", factory="mod:fn", packages=[...])``: the name
      is the first argument; pass exactly one of ``gym=`` (a gym id) or ``factory=``
      (a ``module:callable``). The sugar builds the validated recipe dataclasses.

    Args:
        source: An ``EnvRecipe`` subclass, a ``Recipe``, or a name (with
            ``gym=``/``factory=``).
        gym: A gym id, for the flat form. Mutually exclusive with ``factory``.
        factory: A ``module:callable`` factory, for the flat form.
        packages: Pip packages the env needs (flat form).
        imports: Module names imported for gym registration (flat form, gym only).
        overwrite: Allow replacing an existing registration with the same name.

    Returns:
        The first argument: the class for an ``EnvRecipe`` (decorator-friendly), the
        ``Recipe`` for an object, or the synthesized ``Recipe`` for the flat form.

    Raises:
        ValueError: If the name is already registered and ``overwrite`` is ``False``.
    """
    if is_env_recipe(source):
        _reject_flat_kwargs(gym, factory, packages, imports)
        # An authored EnvRecipe carries its own defining module, so stage a
        # ProjectInstall from there regardless of who calls register().
        store_recipe(
            source.to_recipe(), overwrite=overwrite, origin=class_origin_dir(source)
        )
        return source
    if isinstance(source, Recipe):
        _reject_flat_kwargs(gym, factory, packages, imports)
        return store_recipe(source, overwrite=overwrite, origin=_caller_origin())
    if isinstance(source, str):
        recipe = _flat_recipe(source, gym, factory, packages, imports)
        return store_recipe(recipe, overwrite=overwrite, origin=_caller_origin())
    raise TypeError(
        "register() first argument must be a name string, a Recipe, or an "
        f"EnvRecipe subclass, got {type(source).__name__}"
    )


def _reject_flat_kwargs(
    gym: str | None,
    factory: str | None,
    packages: Sequence[str],
    imports: Sequence[str],
) -> None:
    if gym or factory or packages or imports:
        raise TypeError(
            "register(Recipe | EnvRecipe) takes no gym=/factory=/packages=/"
            "imports=; those are for the flat register(name, ...) form"
        )


def _flat_recipe(
    name: str,
    gym: str | None,
    factory: str | None,
    packages: Sequence[str],
    imports: Sequence[str],
) -> Recipe:
    if (gym is None) == (factory is None):
        raise TypeError(
            f"register({name!r}, ...) needs exactly one of gym= or factory="
        )
    if gym is not None:
        return gym_sugar_to_recipe(name, gym, packages=packages, imports=imports)
    assert factory is not None
    return factory_sugar_to_recipe(name, factory, packages=packages, imports=imports)


def store_recipe(
    recipe: Recipe, *, overwrite: bool, origin: str | None = None
) -> Recipe:
    existing = _REGISTRY.get(recipe.name)
    if existing is not None and not overwrite and existing != recipe:
        raise ValueError(
            f"recipe {recipe.name!r} is already registered; pass overwrite=True to replace it"
        )
    _REGISTRY[recipe.name] = recipe
    if origin is not None:
        _ORIGINS[recipe.name] = origin
    else:
        # A re-registration without a determinable origin must not leave a stale
        # one behind that points at the previous registrant's tree.
        _ORIGINS.pop(recipe.name, None)
    return recipe


def class_origin_dir(cls: type[object]) -> str | None:
    """Return the directory of the module that defines ``cls``, or None.

    Best-effort: a dynamically created class (no source file) has no origin. The
    sandbox uses this to stage an authored ``EnvRecipe`` or ``ModelRecipe``'s
    ProjectInstall from its defining package rather than the launching process's
    current directory.
    """
    try:
        source_file = inspect.getfile(cls)
    except (TypeError, OSError):
        return None
    return str(Path(source_file).resolve().parent)


def _caller_origin() -> str | None:
    """Return the directory of the module that *called* ``register()``, or None.

    Walks past frames in this module so the recorded origin is the registrant's
    own file, not ``_registry.py``. Best-effort: returns None when the caller has
    no file path (e.g. ``__main__`` typed at a REPL, or an exec'd string).
    """
    this_file = __file__
    for frame_info in inspect.stack()[1:]:
        filename = frame_info.filename
        if filename == this_file:
            continue
        module = frame_info.frame.f_globals.get("__name__")
        # A REPL / exec'd snippet has no importable on-disk package to stage from.
        if filename in ("<stdin>", "<string>") or module == "__main__":
            return None
        if not Path(filename).exists():
            return None
        return str(Path(filename).resolve().parent)
    return None


def recipe_origin_dir(name: str) -> str | None:
    """Return the defining-package directory recorded for a registered recipe.

    Best-effort: returns ``None`` when no recipe is registered under ``name`` or
    its origin could not be determined at ``register()`` time. The sandbox uses
    this to stage a ProjectInstall from the recipe author's source tree instead of
    the launching process's current directory.
    """
    return _ORIGINS.get(name)


def from_recipe_origin(recipe: Recipe) -> str | None:
    """Return the origin of the *terminal* ``from_recipe`` base, or None.

    ``resolve_from_recipe`` recurses through a chain of ``from_recipe`` references
    and inlines the TERMINAL base's build (the first ancestor whose
    ``build.from_recipe`` is None) -- and ``from_recipe`` is exclusive with other
    build fields, so that terminal base owns any ProjectInstall, with ``src``
    relative to *its* source tree. Walk the chain from ``recipe.build.from_recipe``
    until the terminal base and return ``recipe_origin_dir(terminal_name)`` so the
    sandbox stages from the right tree, not the immediate base's / cwd.

    Best-effort: returns ``None`` when ``recipe.build.from_recipe`` is None, when a
    referenced base is not registered (let ``resolve_from_recipe`` raise the
    canonical error), or when the terminal base has no recorded origin. Cycles are
    guarded with a seen-set (``resolve_from_recipe`` raises the canonical cycle
    error elsewhere).
    """
    name = recipe.build.from_recipe
    if name is None:
        return None
    seen: set[str] = set()
    while name not in seen:
        seen.add(name)
        try:
            base = resolve(name)
        except RecipeNotFoundError:
            return None
        next_name = base.build.from_recipe
        if next_name is None:
            return recipe_origin_dir(name)
        name = next_name
    return None


def resolve(name: str) -> Recipe:
    """Look up a registered recipe by name.

    Raises:
        RecipeNotFoundError: If no recipe is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise RecipeNotFoundError(
            f"no recipe registered as {name!r}; registered recipes: {available}"
        ) from None


def resolve_from_recipe(recipe: Recipe, _seen: frozenset[str] = frozenset()) -> Recipe:
    """Inline a ``build.from_recipe`` reference into the recipe's build phase.

    Build reuse for an N-task family: a build-only base recipe holds the build,
    and each task recipe references it by name via ``build.from_recipe`` while
    carrying its own ``make``/``setup``. Inlining the base's build into the child
    before the wire makes every task in the family share one ``build_hash`` (one
    image). A recipe with no ``from_recipe`` is returned unchanged.

    Raises:
        ValueError: If ``from_recipe`` is combined with other build fields, or the
            chain of references is cyclic.
        RecipeNotFoundError: If the referenced base is not registered.
    """
    name = recipe.build.from_recipe
    if name is None:
        return recipe
    if recipe.build != Build(from_recipe=name):
        raise ValueError(
            "Build.from_recipe is exclusive with other build fields; the base "
            "recipe supplies the entire build phase"
        )
    if name in _seen:
        raise ValueError(f"from_recipe reference cycle through {name!r}")
    base = resolve_from_recipe(resolve(name), _seen | {name})
    return dataclasses.replace(recipe, build=base.build)


def registered_names() -> tuple[str, ...]:
    """Return the sorted names of all locally registered recipes."""
    return tuple(sorted(_REGISTRY))


def registry() -> Mapping[str, Recipe]:
    """Return a read-only view of the local ``name -> recipe`` map.

    The view reflects later registrations; it cannot be mutated (use
    :func:`register`/:func:`unregister`).
    """
    return MappingProxyType(_REGISTRY)


def _make_kind(recipe: Recipe) -> str:
    make = recipe.make
    if make is None:
        return "base"  # a build-only base (e.g. a from_recipe parent)
    if isinstance(make, GymMake):
        return "gym"
    if isinstance(make, PyMake):
        return "py"
    return "hf"  # Make is a closed union; the remaining arm is HfMake


def _format_registry() -> str:
    names = registered_names()
    header = f"===== rlmesh recipes ({len(names)}) ====="
    if not names:
        return f"{header}\n  <empty>"

    groups: dict[str, list[str]] = {}
    for name in names:
        namespace = name.split("/", 1)[0] if "/" in name else ""
        groups.setdefault(namespace, []).append(name)

    name_width = max(len(name) for name in names)
    lines = [header]
    for namespace in sorted(groups):
        lines.append(f"{namespace}/" if namespace else "(root)")
        for name in groups[namespace]:
            recipe = _REGISTRY[name]
            kind = _make_kind(recipe)
            gpu = "gpu" if recipe.build.gpu else ""
            summary = recipe.summary or "—"
            lines.append(f"  {name:<{name_width}}  {kind:<4} {gpu:<3}  {summary}")
    return "\n".join(lines)


def pprint_registry(*, disable_print: bool = False) -> str | None:
    """Pretty-print the registered recipes, grouped by namespace.

    Shows each recipe's name, make kind (``gym``/``py``/``hf``/``base``), a ``gpu``
    marker, and its summary. Covers only the local rlmesh recipe registry, not the
    Gymnasium registry (use ``gymnasium.pprint_registry()`` for that).

    Args:
        disable_print: Return the formatted string instead of printing it.
    """
    text = _format_registry()
    if disable_print:
        return text
    print(text)
    return None


def unregister(name: str) -> None:
    """Remove a recipe from the registry if present (no error when absent)."""
    _REGISTRY.pop(name, None)
    _ORIGINS.pop(name, None)


def clear_registry() -> None:
    """Remove every registered recipe (primarily for test isolation)."""
    _REGISTRY.clear()
    _ORIGINS.clear()
