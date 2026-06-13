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

from ._schema import Recipe

__all__ = [
    "RecipeNotFoundError",
    "clear_registry",
    "register",
    "registered_names",
    "resolve",
    "unregister",
]


class RecipeNotFoundError(LookupError):
    """Raised when a recipe name is not present in the local registry."""


_REGISTRY: dict[str, Recipe] = {}


def register(recipe: Recipe, *, overwrite: bool = False) -> Recipe:
    """Register a recipe under its ``name`` in the process-local registry.

    Args:
        recipe: The recipe to register.
        overwrite: Allow replacing an existing registration with the same name.

    Returns:
        The registered recipe (so ``register`` can wrap a literal at module load).

    Raises:
        ValueError: If ``recipe.name`` is already registered and ``overwrite`` is
            ``False`` (anti-shadowing).
    """
    existing = _REGISTRY.get(recipe.name)
    if existing is not None and not overwrite and existing != recipe:
        raise ValueError(
            f"recipe {recipe.name!r} is already registered; pass overwrite=True to replace it"
        )
    _REGISTRY[recipe.name] = recipe
    return recipe


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


def registered_names() -> tuple[str, ...]:
    """Return the sorted names of all locally registered recipes."""
    return tuple(sorted(_REGISTRY))


def unregister(name: str) -> None:
    """Remove a recipe from the registry if present (no error when absent)."""
    _REGISTRY.pop(name, None)


def clear_registry() -> None:
    """Remove every registered recipe (primarily for test isolation)."""
    _REGISTRY.clear()
