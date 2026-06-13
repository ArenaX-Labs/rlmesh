"""Environment recipes: inert, JSON-round-trippable construction documents.

A recipe declares how to *construct* an environment (build a Dockerfile, set up
construct-time data, and name a factory) -- distinct from an adapter, which only
describes IO. See ``rlmesh.recipes._schema`` for the three-phase schema.

The authoring tiers:

* Tier 0 -- a bare string id, never leaves ``rlmesh.make("CartPole-v1")``.
* Tier 1 -- ``rlmesh.recipes.*`` typed recipes, opened when a string is not enough.
"""

from __future__ import annotations

from ._build import build
from ._launch import (
    SandboxLaunchArgs,
    UnsupportedRecipeError,
    recipe_to_sandbox_args,
)
from ._make import make
from ._registry import (
    RecipeNotFoundError,
    clear_registry,
    register,
    registered_names,
    resolve,
    unregister,
)
from ._schema import (
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
    RecipeValidationError,
    Requires,
    Setup,
)

__all__ = [
    "Build",
    "Fetch",
    "FileWrite",
    "GymMake",
    "HfMake",
    "Make",
    "PipInstall",
    "ProjectInstall",
    "PyMake",
    "Recipe",
    "RecipeNotFoundError",
    "RecipeValidationError",
    "Requires",
    "SandboxLaunchArgs",
    "Setup",
    "UnsupportedRecipeError",
    "build",
    "clear_registry",
    "make",
    "recipe_to_sandbox_args",
    "register",
    "registered_names",
    "resolve",
    "unregister",
]
