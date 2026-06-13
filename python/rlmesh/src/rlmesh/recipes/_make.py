"""``rlmesh.make`` -- a strict superset of ``gymnasium.make``.

``make`` resolves its first argument (a registered recipe name, a literal
``Recipe``, or a bare gym id) and constructs the environment in the current
process. The build phase is *dormant locally* (advisory only -- it never runs
apt/git in-process); a recipe that genuinely needs its image should be launched
through ``SandboxEnv``. A bare id that is not a registered recipe falls through
to ``gymnasium.make`` (recipes first, gym fallthrough; the gym registry is a
zero-config recipe registry).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ._build import build
from ._registry import RecipeNotFoundError, resolve
from ._schema import GymMake, Recipe

if TYPE_CHECKING:
    from rlmesh.server import EnvLike

__all__ = ["make"]


def make(
    source: str | Recipe,
    *,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    **kwargs: object,
) -> EnvLike:
    """Construct an environment from a recipe name, a ``Recipe``, or a gym id.

    Args:
        source: A registered recipe name, a literal ``Recipe``, or a gym id.
        num_envs: Number of environment instances to create.
        vectorization_mode: Vectorization mode for ``num_envs > 1``.
        **kwargs: Extra factory kwargs, forwarded to ``gymnasium.make`` for the
            gym-id fallthrough only.

    Returns:
        The constructed environment.
    """
    recipe = _coerce_recipe(source, kwargs)
    return build(recipe, num_envs=num_envs, vectorization_mode=vectorization_mode)


def _coerce_recipe(source: str | Recipe, kwargs: Mapping[str, object]) -> Recipe:
    if isinstance(source, Recipe):
        if kwargs:
            raise TypeError(
                "make(recipe) does not accept extra kwargs; bake them into the recipe"
            )
        return source
    try:
        recipe = resolve(source)
    except RecipeNotFoundError:
        # Gym fallthrough: rlmesh.make is a strict superset of gymnasium.make.
        return Recipe(
            name=f"gym/{source}", make=GymMake(env_id=source, kwargs=dict(kwargs))
        )
    if kwargs:
        raise TypeError(
            f"make({source!r}) resolved to a registered recipe; extra kwargs are not "
            "supported (bake them into the recipe)"
        )
    return recipe
