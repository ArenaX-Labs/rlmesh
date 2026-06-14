"""Environment recipes: inert, JSON-round-trippable construction documents.

A recipe declares how to *construct* an environment (build a Dockerfile, set up
construct-time data, and name a factory) -- distinct from an adapter, which only
describes how observations and actions map. The headline authoring surface is
:class:`EnvRecipe` (subclass it);
:class:`Recipe` and the ``Build``/``Make`` dataclasses are the inert form your
recipe lowers to. See ``rlmesh.recipes._schema`` for the three-phase schema.

Tiers:

* Top level (``rlmesh.X``): ``make``, ``register``, ``EnvRecipe``, ``Recipe``.
* This module (``rlmesh.recipes.X``): the build vocabulary you opt into for heavy
  envs -- ``Build``, ``PipInstall``, ``Fetch``, ``ProjectInstall``, ``Setup``,
  ``GymMake``/``PyMake``/``HfMake`` -- plus ``register``/``resolve``/``check``.
* ``rlmesh.recipes.scaffold``: one-shot migration tooling.
"""

from __future__ import annotations

from rlmesh._bootstrap.env import RecipeConstructionError

from ._artifacts import hf_load, input_path
from ._authoring import EnvRecipe
from ._authoring_model import ModelRecipe
from ._build import build
from ._check import check
from ._launch import (
    SandboxLaunchArgs,
    UnsupportedRecipeError,
    recipe_to_sandbox_args,
)
from ._make import make
from ._registry import (
    RecipeNotFoundError,
    clear_registry,
    pprint_registry,
    register,
    registered_names,
    registry,
    resolve,
    resolve_from_recipe,
    unregister,
)
from ._schema import (
    ArtifactInput,
    Build,
    Fetch,
    FileWrite,
    GymMake,
    HfMake,
    Make,
    PipInstall,
    ProjectInstall,
    PyMake,
    Recipe,
    RecipeKind,
    RecipeValidationError,
    Requires,
    RuntimeReserved,
    Setup,
)

__all__ = [
    "ArtifactInput",
    "Build",
    "EnvRecipe",
    "Fetch",
    "FileWrite",
    "GymMake",
    "HfMake",
    "Make",
    "ModelRecipe",
    "PipInstall",
    "ProjectInstall",
    "PyMake",
    "Recipe",
    "RecipeConstructionError",
    "RecipeKind",
    "RecipeNotFoundError",
    "RecipeValidationError",
    "Requires",
    "RuntimeReserved",
    "SandboxLaunchArgs",
    "Setup",
    "UnsupportedRecipeError",
    "build",
    "check",
    "clear_registry",
    "hf_load",
    "input_path",
    "make",
    "pprint_registry",
    "recipe_to_sandbox_args",
    "register",
    "registered_names",
    "registry",
    "resolve",
    "resolve_from_recipe",
    "unregister",
]
