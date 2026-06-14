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
import warnings
from typing import TYPE_CHECKING, cast

from rlmesh._bootstrap.env import (
    import_gym_modules,
    import_packages,
    is_env_lookup_error,
    load_env_entrypoint,
    make_gym_environment,
)

from ._launch import UnsupportedRecipeError
from ._schema import GymMake, PyMake, Recipe, Setup

if TYPE_CHECKING:
    from rlmesh.server import EnvLike

__all__ = ["build"]


def apply_setup(setup: Setup) -> None:
    """Apply construct-time setup data (phase 2).

    ``setup.env`` mutates ``os.environ`` best-effort -- it is intentionally not
    isolation-safe (constructed envs read vars lazily, so a restore would break
    the live env); the sandbox is the blessed isolation path. ``setup.files`` is
    not yet applied *anywhere* (the tempdir-gated file writer has not landed), so a
    recipe carrying it is rejected here; the sandbox bootstrap calls this same
    ``apply_setup``, so running in a sandbox does not work around it either.
    """
    for key, value in setup.env.items():
        os.environ[key] = value
    if setup.files:
        raise UnsupportedRecipeError(
            "setup.files is not applied yet (local or sandbox); remove it and stage "
            "files via the build phase (build.project / build.fetch) instead"
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
        vectorization_mode: Vectorization mode for ``num_envs > 1``. Ignored when
            ``num_envs == 1`` (a single env is not vectorized) -- the in-container
            bootstrap always passes ``"sync"`` for a single env.

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

    apply_setup(recipe.setup)

    if isinstance(make, GymMake):
        # Registration side effects -- gym only.
        import_packages(recipe.requires.imports)
        gym_modules = import_gym_modules()
        if not gym_modules:
            raise ImportError(
                "gymnasium or gym must be installed to build a gym recipe"
            )
        # Mirror load_gym_env: try every module in preference order, moving on when
        # one cannot find the env id, so an env registered only in legacy gym still
        # resolves when gymnasium is also installed.
        errors: list[tuple[str, Exception]] = []
        env = None
        for gym_module in gym_modules:
            try:
                env = make_gym_environment(
                    gym_module,
                    env_id=make.env_id,
                    kwargs=dict(make.kwargs),
                    num_envs=num_envs,
                    vectorization_mode=vectorization_mode,
                )
                break
            except Exception as exc:
                if is_env_lookup_error(exc):
                    errors.append((getattr(gym_module, "__name__", "<unknown>"), exc))
                    continue
                raise
        if env is None:
            names = ", ".join(name for name, _ in errors)
            raise RuntimeError(
                f"failed to create gym environment {make.env_id!r} with {names}"
            ) from (errors[0][1] if errors else None)
    elif isinstance(make, PyMake):
        # NO pre-import: the factory body is the sole import sequencer. The loader
        # imports only the (empty) package list, then resolves and calls the factory.
        # A single env (num_envs == 1) is fine regardless of vectorization_mode --
        # there is no vectorization with one env, and the bootstrap always passes
        # vectorization_mode="sync". Only a genuine vector request is rejected.
        if num_envs != 1:
            raise TypeError(
                "num_envs/vectorization_mode apply to gym sources only; a py factory "
                "returns one env -- vectorize inside the factory"
            )
        env = load_env_entrypoint(make.entrypoint, kwargs=dict(make.kwargs))
    else:  # HfMake -- its source materializes only inside a sandbox.
        raise UnsupportedRecipeError(
            f"recipe {recipe.name!r} uses HfMake, which materializes its source only "
            "inside a sandbox; run it via the sandbox path"
        )

    if recipe.adapter is not None and recipe.kind == "env":
        _publish_env_tags(env, recipe.adapter)

    return cast("EnvLike", env)


def _publish_env_tags(env: object, adapter: object) -> None:
    """Publish a recipe-launched env's :class:`EnvTags` (the forward path).

    This runs ``join()`` against the env's real spaces and fails loud. The adapters
    layer is resolved dynamically so the recipe layer does not hard-depend on it.
    A missing ``rlmesh.adapters`` is warned about once and skipped, so a recipe with
    an adapter block still constructs; errors raised by ``tag()`` itself (once adapters
    *is* importable) still propagate.

    Note: ``EnvRecipe.to_recipe()`` does not populate ``recipe.adapter`` -- env tags
    normally ride ``tag()``/``EnvServer(tags=)`` -- so this path is usually inert, but
    the read site must be correct and crash-free.
    """
    import importlib

    try:
        adapters = importlib.import_module("rlmesh.adapters")
    except ImportError:
        warnings.warn(
            "rlmesh.adapters is not available; skipping recipe env-tag publishing "
            "(env tags will be enforced once the adapters layer lands)",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    env_tags_cls = adapters.EnvTags
    # ``recipe.adapter`` is a raw JSON Mapping after from_dict, or an EnvTags instance
    # when constructed in-process. Rehydrate the former before publishing.
    tags = adapter if isinstance(adapter, env_tags_cls) else env_tags_cls.from_dict(adapter)
    adapters.tag(env, tags)
