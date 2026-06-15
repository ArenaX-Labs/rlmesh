"""Build a recipe into a self-describing Docker image."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike, fspath
from types import MappingProxyType
from typing import TYPE_CHECKING, TypedDict, cast

from .._rlmesh import sandbox_build_image as _sandbox_build_image

if TYPE_CHECKING:
    from ..recipes import ArtifactInput, EnvRecipe, Recipe

# Frozen empty default (avoids the mutable-default trap) for callers passing no kwargs.
_NO_MAKE_KWARGS: Mapping[str, object] = MappingProxyType({})


class _SandboxBuildInfo(TypedDict):
    requested_source: str
    resolved_source: str
    image: str
    alias: str | None
    image_id: str


@dataclass(frozen=True)
class ExportResult:
    """The image produced by :func:`export`.

    ``image`` is the deterministic, content-addressed reference; it is stable
    for a given build, so the managed platform pins it. ``alias`` is the human
    tag passed as ``tag=``, if any. Both name the same image.
    """

    requested_source: str
    resolved_source: str
    image: str
    alias: str | None
    image_id: str


def export(
    source: str | Recipe | type[EnvRecipe] | type,
    *,
    tag: str | None = None,
    base_image: str | None = None,
    rlmesh_package: str | PathLike[str] | None = None,
    packages: Sequence[str] = (),
    trust_remote_code: bool = False,
    allow_unpinned_hf: bool = False,
    build_memory: str | None = None,
) -> ExportResult:
    """Build a recipe into a Docker image and return its reference.

    No container is started. The image is self-describing -- it bakes the recipe
    document and the kind-aware entrypoint, so ``docker run`` with no arguments
    serves the env or model on port 50051. It is the same image the RLMesh Managed
    platform runs; ``docker push`` the returned reference to a registry it can reach.

    Works for both env recipes (``EnvRecipe``, a ``Recipe``, or a registered env
    name) and model recipes (``ModelRecipe``, a ``kind='model'`` Recipe, or a
    registered model name); the kind selects the baked entrypoint. ``tag`` adds a
    human alias alongside the always-applied content-addressed
    ``rlmesh-sandbox-<slug>:<hash>`` tag.
    """
    from .session import string_sequence

    if _is_model_source(source):
        from ._model import resolve_model_recipe

        recipe, context_root = resolve_model_recipe(source)
        display, recipe_json, provenance = recipe.name, recipe.to_json(), "installed"
    else:
        display, recipe_json, provenance, context_root, _inputs = resolve_recipe_source(
            source, {}, ()
        )
    info = cast(
        _SandboxBuildInfo,
        _sandbox_build_image(
            display,
            tag=tag,
            base_image=base_image,
            rlmesh_package=normalize_rlmesh_package(rlmesh_package),
            packages=string_sequence("packages", packages),
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            recipe_json=recipe_json,
            recipe_provenance=provenance,
            context_root=context_root,
            build_memory=build_memory,
        ),
    )
    return ExportResult(**info)


def _is_model_source(source: object) -> bool:
    """Whether ``source`` names a model recipe (vs an env recipe)."""
    from ..recipes import Recipe, resolve
    from ..recipes._registry import RecipeNotFoundError
    from ..recipes.authoring.model import is_model_recipe

    if is_model_recipe(source):
        return True
    if isinstance(source, Recipe):
        return source.kind == "model"
    if isinstance(source, str):
        # An unregistered name is not a model here; let the env path raise.
        try:
            return resolve(source).kind == "model"
        except RecipeNotFoundError:
            return False
    return False


def resolve_recipe_source(
    source: str | Recipe | type[EnvRecipe],
    gym_make_kwargs: Mapping[str, object] = _NO_MAKE_KWARGS,
    imports: Sequence[str] | None = None,
) -> tuple[str, str | None, str | None, str | None, tuple[ArtifactInput, ...]]:
    """Resolve a sandbox source into (display, recipe_json, provenance, context_root, inputs).

    An ``EnvRecipe`` subclass or an in-process ``Recipe`` is ``Installed`` -- it
    came from your installed/loaded code (pip-install-is-consent), so its build
    (including ``ProjectInstall``) is trusted. A registered name is ``Installed``
    too. ``Remote`` is reserved for a document handed in from an untrusted external
    source (the future catalog/wire path). A plain id/name that is not a recipe is
    an ordinary gym/hf source string, unchanged. When the recipe stages a project
    tree, ``context_root`` is the recipe's defining-package directory when that can
    be determined, falling back to the current directory.

    ``gym_make_kwargs`` are baked into ``recipe.make.kwargs`` so the in-container
    ``build()`` forwards them to the factory, matching local
    ``rlmesh.make(recipe, **kwargs)``. They are *not* also forwarded via
    ``kwargs_json`` on the recipe path (the recipe bootstrap payload carries only
    ``make.kwargs``), so nothing is applied twice.

    ``imports`` are merged into ``recipe.requires.imports`` for the same reason: the
    recipe bootstrap reads ``requires.imports`` in-container, never the caller's
    ``imports=`` (which only the gym/hf path forwards). Merging keeps a caller's
    registration import (e.g. ``ale_py``) from being silently dropped on the recipe
    path. ``requires.imports`` is meaningless for a ``PyMake``/build-only recipe (the
    py factory owns its own imports), so caller imports on those raise.
    """
    import dataclasses
    import os

    from ..recipes import (
        HfMake,
        PyMake,
        RecipeNotFoundError,
        UnsupportedRecipeError,
        resolve,
        resolve_from_recipe,
    )
    from ..recipes._registry import (
        class_origin_dir,
        from_recipe_origin,
        recipe_origin_dir,
    )
    from ..recipes.authoring.env import as_authored_recipe, is_env_recipe

    # ``origin`` is the filesystem directory of the code that *defined* the recipe,
    # used to stage a ProjectInstall from the package's own source tree rather than
    # the caller's cwd. ``None`` means we could not determine it.
    origin: str | None = None
    authored = as_authored_recipe(source)
    if authored is not None:
        recipe = authored
        provenance = "installed"
        if is_env_recipe(source):
            # An authored EnvRecipe knows its defining module; stage from there.
            origin = class_origin_dir(source)
    elif isinstance(source, str):
        try:
            recipe = resolve(source)
        except RecipeNotFoundError:
            return source, None, None, None, ()
        provenance = "installed"
        # A name resolves to a recipe registered by some package; prefer that
        # registrant's module directory when the registry recorded it.
        origin = recipe_origin_dir(source)
    else:
        raise TypeError(
            f"sandbox source must be a str, Recipe, or EnvRecipe, got "
            f"{type(source).__name__}"
        )
    # Capture the terminal-base origin BEFORE inlining: an inlined ProjectInstall's
    # `src` is relative to that base's tree, so we stage from it below.
    from_recipe_base_origin = from_recipe_origin(recipe)
    # Inline any `from_recipe` base build so a task family shares one image.
    recipe = resolve_from_recipe(recipe)
    # An HfMake recipe materializes its source only via the sandbox HF path, never
    # the recipe path; reject it here so we fail fast instead of after a full image
    # build (the in-container build() would otherwise raise UnsupportedRecipeError).
    if isinstance(recipe.make, HfMake):
        raise UnsupportedRecipeError(
            f"recipe {recipe.name!r} uses HfMake, which materializes its source only "
            "via the sandbox HF path, not the recipe path; pass the HF source string "
            "to SandboxEnv instead"
        )
    # setup.files is not applied anywhere yet (the tempdir-gated file writer has not
    # landed; the in-container build() -> apply_setup raises on it too). Reject it
    # here so we fail fast instead of after a full image build.
    if recipe.setup.files:
        raise UnsupportedRecipeError(
            "setup.files is not applied yet (local or sandbox); remove it and stage "
            "files via the build phase (build.project / build.fetch) instead"
        )
    # Bake make kwargs into the document; a build-only base has no make to carry them.
    if gym_make_kwargs:
        if recipe.make is None:
            raise TypeError(
                f"recipe {recipe.name!r} is a build-only base (make is None); it takes "
                "no environment make kwargs"
            )
        merged_make = dataclasses.replace(
            recipe.make, kwargs={**recipe.make.kwargs, **gym_make_kwargs}
        )
        recipe = dataclasses.replace(recipe, make=merged_make)
    # Merge caller imports into requires.imports (the bootstrap reads only that). A
    # PyMake/build-only recipe owns its own imports, so reject rather than drop them.
    if imports:
        if recipe.make is None or isinstance(recipe.make, PyMake):
            raise TypeError(
                f"recipe {recipe.name!r} is a PyMake/build-only recipe; imports= does "
                "not apply -- a py factory performs its own imports"
            )
        # De-dup, preserving order: the recipe's own imports first.
        merged_imports = list(recipe.requires.imports)
        for name in imports:
            if name not in merged_imports:
                merged_imports.append(name)
        merged_requires = dataclasses.replace(recipe.requires, imports=merged_imports)
        recipe = dataclasses.replace(recipe, requires=merged_requires)
    if recipe.build.project is None:
        context_root = None
    else:
        # For a from_recipe chain the inlined project.src is relative to the terminal
        # base's tree, so prefer its origin. Stage from the recipe's defining package
        # when known, else the caller's cwd (a Recipe assembled at the call site has
        # no determinable origin).
        if from_recipe_base_origin is not None:
            origin = from_recipe_base_origin
        context_root = origin if origin is not None else os.getcwd()
    return recipe.name, recipe.to_json(), provenance, context_root, recipe.inputs


def normalize_rlmesh_package(value: str | PathLike[str] | None) -> str | None:
    if value is None:
        return None
    return fspath(value)


__all__ = ["ExportResult", "export"]
