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

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from ._construct import build
from ._gym_sugar import gym_sugar_to_recipe
from ._registry import RecipeNotFoundError, resolve
from ._schema import Recipe
from .authoring.env import EnvRecipe, construct_authored, is_env_recipe

if TYPE_CHECKING:
    from rlmesh.server import EnvLike

__all__ = ["make"]


def make(
    source: str | Recipe | type[EnvRecipe],
    *,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
    packages: Sequence[str] | None = None,
    imports: Sequence[str] | None = None,
    **kwargs: object,
) -> EnvLike:
    """Construct an environment from a name, ``Recipe``, ``EnvRecipe``, or gym id.

    Args:
        source: A registered recipe name, a ``Recipe``, an ``EnvRecipe`` subclass,
            or a gym id.
        num_envs: Number of environment instances to create (gym sources only).
        vectorization_mode: Vectorization mode for ``num_envs > 1``.
        packages: For a gym id, the pip packages the env needs. Recorded on the
            recipe and installed in a sandbox; **advisory locally** (the build phase
            is dormant in-process, so the packages must already be importable).
        imports: For a gym id, module names imported so the env registers itself
            (e.g. ``["ale_py"]``). These ARE imported in-process.
        **kwargs: Extra factory kwargs (forwarded to ``gymnasium.make`` for a gym id).

    Returns:
        The constructed environment.
    """
    if isinstance(source, EnvRecipe):
        raise TypeError(
            "pass the EnvRecipe subclass, not an instance: rlmesh.make(MyEnv) not "
            "rlmesh.make(MyEnv())"
        )
    if is_env_recipe(source):
        # Construct in-process via the class lifecycle (works for a class defined
        # in a script/notebook). The gym sugar and vectorization do not apply.
        _reject_unsupported_for_class(packages, imports, num_envs, vectorization_mode)
        return construct_authored(source, **kwargs)
    recipe = _coerce_recipe(cast("str | Recipe", source), kwargs, packages, imports)
    return build(recipe, num_envs=num_envs, vectorization_mode=vectorization_mode)


def _reject_unsupported_for_class(
    packages: Sequence[str] | None,
    imports: Sequence[str] | None,
    num_envs: int,
    vectorization_mode: str | None,
) -> None:
    if packages or imports:
        raise TypeError(
            "make(EnvRecipe) does not accept packages=/imports=; declare them in the "
            "class's build= (and import inside make())"
        )
    if num_envs != 1 or vectorization_mode is not None:
        raise TypeError(
            "make(EnvRecipe) does not support num_envs/vectorization_mode (a py factory "
            "returns one env); vectorize inside your make() or use a gym source"
        )


def _coerce_recipe(
    source: str | Recipe,
    kwargs: Mapping[str, object],
    packages: Sequence[str] | None,
    imports: Sequence[str] | None,
) -> Recipe:
    if isinstance(source, Recipe):
        if kwargs or packages or imports:
            raise TypeError(
                "make(recipe) does not accept extra kwargs/packages/imports; bake "
                "them into the recipe"
            )
        return source
    try:
        recipe = resolve(source)
    except RecipeNotFoundError:
        # Gym fallthrough: rlmesh.make is a strict superset of gymnasium.make. The
        # synthesized (throwaway) recipe name is sanitized -- a gym id may contain
        # ':' (e.g. sai_pygame:SquidHunt-v0), which a recipe name may not.
        name = "gym/" + re.sub(r"[^A-Za-z0-9._/-]", "-", source)
        return gym_sugar_to_recipe(
            name, source, packages=packages, imports=imports, kwargs=kwargs
        )
    if kwargs or packages or imports:
        raise TypeError(
            f"make({source!r}) resolved to a registered recipe; extra "
            "kwargs/packages/imports are not supported (bake them into the recipe)"
        )
    return recipe
