"""Construct an environment from a recipe (phases 2 + 3, in-process).

``build`` orchestrates the construct-time half of a recipe: it applies ``setup``
(phase 2) and then calls the named ``make`` factory (phase 3). Phase 1 (the
Dockerfile) has already run by the time this executes inside a sandbox; locally
it is skipped (the env's dependencies must already be importable).

The keystone is the ``PyMake`` branch: it dispatches through the existing
``load_env_entrypoint`` -- a loader that has always existed but was never
reachable, because the bootstrap had no ``kind="recipe"``. Critically, the py
path does **not** pre-import ``requires.imports``: the factory body owns its own
import sequence (an Isaac ``SimulationApp`` must be created before
``isaaclab.*`` imports), so an envelope-level pre-import would crash exactly the
environments this path targets. The gym/hf paths still run ``requires.imports``
for side-effect registration.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from rlmesh._bootstrap.env import (
    import_gym_modules,
    import_packages,
    load_env_entrypoint,
    make_gym_environment,
)

from ._launch import UnsupportedRecipeError
from ._schema import GymMake, PyMake, Recipe, Setup

if TYPE_CHECKING:
    from rlmesh.server import EnvLike

__all__ = ["build"]


def _apply_setup(setup: Setup) -> None:
    """Apply construct-time setup data (phase 2).

    ``setup.env`` mutates ``os.environ`` best-effort -- it is intentionally not
    isolation-safe (constructed envs read vars lazily, so a restore would break
    the live env); the sandbox is the blessed isolation path. ``setup.files`` is
    not yet applied in-process (it lands with the tempdir-gated file writer); a
    recipe that needs it should run in a sandbox.
    """
    for key, value in setup.env.items():
        os.environ[key] = value
    if setup.files:
        raise UnsupportedRecipeError(
            "setup.files is not applied in-process yet; run the recipe in a sandbox"
        )


def build(
    recipe: Recipe,
    *,
    num_envs: int = 1,
    vectorization_mode: str | None = None,
) -> EnvLike:
    """Construct an environment from a recipe, in the current process.

    Args:
        recipe: The recipe to construct. Must not be a build-only base.
        num_envs: Number of environment instances to create.
        vectorization_mode: Vectorization mode for ``num_envs > 1``.

    Returns:
        The constructed environment.

    Raises:
        ValueError: If ``recipe.make`` is ``None`` (a build-only base).
        UnsupportedRecipeError: For an ``HfMake`` recipe, which materializes its
            source only inside a sandbox.
    """
    make = recipe.make
    if make is None:
        raise ValueError(
            f"{recipe.name!r} is a build-only base recipe; reference it via from_recipe, "
            "do not construct it"
        )

    _apply_setup(recipe.setup)

    if isinstance(make, GymMake):
        # Registration side effects -- gym only.
        import_packages(recipe.requires.imports)
        gym_modules = import_gym_modules()
        if not gym_modules:
            raise ImportError(
                "gymnasium or gym must be installed to build a gym recipe"
            )
        env = make_gym_environment(
            gym_modules[0],
            env_id=make.env_id,
            kwargs=dict(make.kwargs),
            num_envs=num_envs,
            vectorization_mode=vectorization_mode,
        )
    elif isinstance(make, PyMake):
        # NO pre-import: the factory body is the sole import sequencer. The loader
        # imports only the (empty) package list, then resolves and calls the factory.
        env = load_env_entrypoint(make.entrypoint, kwargs=dict(make.kwargs))
    else:  # HfMake -- its source materializes only inside a sandbox.
        raise UnsupportedRecipeError(
            f"recipe {recipe.name!r} uses HfMake, which materializes its source only "
            "inside a sandbox; run it via the sandbox path"
        )

    if recipe.annotations is not None:
        _publish_annotations(env, recipe.annotations)

    return cast("EnvLike", env)


def _publish_annotations(env: object, annotations: Mapping[str, object]) -> None:
    """Publish adapter annotations for a recipe-launched env (forward path).

    Registry-spec section 11: this runs ``join()`` against the env's real spaces
    and fails loud. The adapters layer is resolved dynamically so the recipe layer
    does not hard-depend on it (it lives in a separate, later-merged module).
    """
    import importlib

    adapters = importlib.import_module("rlmesh.adapters")
    annotate = adapters.annotate
    annotate(env, annotations)
