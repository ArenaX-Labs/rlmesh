"""Migration scaffolder -- one-shot tooling, kept off the core authoring surface.

Generate a recipe (and a factory stub) from an existing project's metadata. This
is orthogonal to authoring (you use it once when porting an env), so it lives in
its own submodule rather than the flat ``rlmesh.recipes`` namespace.
"""

from __future__ import annotations

from ._scaffold import ScaffoldResult, scaffold_from_pyproject, scaffold_recipe

__all__ = ["ScaffoldResult", "scaffold_from_pyproject", "scaffold_recipe"]
