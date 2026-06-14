"""Validate a recipe without importing its dependencies (authoring != running)."""

from __future__ import annotations

from rlmesh._bootstrap.entrypoint import parse_entrypoint

from ._schema import PyMake, Recipe, RecipeValidationError

__all__ = ["check"]


def check(recipe: Recipe) -> None:
    """Validate a recipe WITHOUT importing its dependencies.

    Round-trips the recipe through JSON and validates the ``make`` entrypoint's
    *string shape* (``module:callable``) -- it never imports the factory. This
    proves a recipe is well-formed on a machine that cannot run it: it catches a
    malformed entrypoint in milliseconds instead of after a long image build. It
    cannot catch a well-shaped-but-wrong entrypoint (that requires importing it);
    a sandbox build is the check for that.

    Raises:
        RecipeValidationError: If the recipe does not round-trip through JSON or its
            entrypoint string is malformed.
    """
    if Recipe.from_json(recipe.to_json()) != recipe:
        raise RecipeValidationError("recipe does not round-trip through JSON")
    if isinstance(recipe.make, PyMake):
        try:
            parse_entrypoint(recipe.make.entrypoint, label="recipe make entrypoint")
        except ValueError as exc:
            raise RecipeValidationError(str(exc)) from exc
