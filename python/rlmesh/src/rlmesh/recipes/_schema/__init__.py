"""The three-phase environment Recipe schema (inert, JSON-round-trippable data).

Split across :mod:`._definitions` (the typed dataclasses) and :mod:`._serialize`
(canonical JSON to_dict/from_dict). This package re-exports the public schema
surface so ``rlmesh.recipes._schema`` stays a stable import path.
"""

from __future__ import annotations

from ._definitions import (
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
    "Fetch",
    "FileWrite",
    "GymMake",
    "HfMake",
    "Make",
    "PipInstall",
    "ProjectInstall",
    "PyMake",
    "Recipe",
    "RecipeKind",
    "RecipeValidationError",
    "Requires",
    "RuntimeReserved",
    "Setup",
]
