"""``SandboxModel``: a model recipe running in its own container.

The model-side sibling of ``SandboxEnv``. The recipe builds to an image whose
ENTRYPOINT is the model bootstrap (the kind-aware deriver selects it), and the
container serves the policy as a model endpoint. Given an env address it drives
that env instead and reports the run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._export import normalize_rlmesh_package

if TYPE_CHECKING:
    from ..recipes import Recipe
    from ..recipes._schema import ArtifactInput

__all__ = ["SandboxModel"]


def resolve_model_recipe(source: object) -> tuple[Recipe, str | None]:
    from ..recipes import Recipe, resolve
    from ..recipes._registry import class_origin_dir, recipe_origin_dir
    from ..recipes._schema import PyMake
    from ..recipes.authoring.model import as_authored_model_recipe, is_model_recipe

    context_root: str | None = None
    if isinstance(source, str):
        recipe = resolve(source)
        context_root = recipe_origin_dir(source)
    elif isinstance(source, Recipe):
        recipe = source
    else:
        recipe = as_authored_model_recipe(source)
        if is_model_recipe(source):
            context_root = class_origin_dir(source)
    if recipe is None or recipe.kind != "model":
        raise TypeError(
            "SandboxModel requires a model recipe (a ModelRecipe subclass, a "
            "kind='model' Recipe, or a registered model name)"
        )
    # A flat registration (register(name, hf=...|load=...)) synthesizes its loader
    # class at register() time and binds it onto this module in the current
    # interpreter only. A fresh container imports the module from disk and never
    # sees that class, so the bootstrap would fail late with an opaque ImportError.
    # Fail early, on the host, with the actionable fix.
    if isinstance(recipe.make, PyMake) and recipe.make.entrypoint.split(":", 1)[0] == (
        "rlmesh.models._registry"
    ):
        raise TypeError(
            f"model {recipe.name!r} is flat-registered (register(name, hf=...|load=...)), "
            "which is in-process only and cannot be launched in a container. "
            "Subclass rlmesh.ModelRecipe so its loader lives in an importable module."
        )
    return recipe, context_root


class SandboxModel:
    """A model recipe started in an isolated container, serving a model endpoint."""

    def __init__(
        self,
        source: object,
        *,
        base_image: str | None = None,
        rlmesh_package: str | None = None,
        packages: Sequence[str] = (),
        artifacts: Sequence[ArtifactInput] = (),
        trust_remote_code: bool = False,
        allow_unpinned_hf: bool = False,
        build_memory: str | None = None,
    ) -> None:
        import json

        from .._rlmesh import sandbox_start_env
        from ..recipes._artifacts import local_dir_mounts

        recipe, context_root = resolve_model_recipe(source)
        # A declared input with a host local_dir is bind-mounted at its container
        # target; uri-backed inputs still resolve in-container. artifacts= supplies
        # or overrides a local_dir by name (the run-time checkpoint).
        mounts = local_dir_mounts(recipe.inputs, artifacts)
        info = sandbox_start_env(
            recipe.name,
            recipe_json=recipe.to_json(),
            recipe_provenance="installed",
            base_image=base_image,
            rlmesh_package=normalize_rlmesh_package(rlmesh_package),
            packages=list(packages),
            trust_remote_code=trust_remote_code,
            allow_unpinned_hf=allow_unpinned_hf,
            context_root=context_root,
            mounts_json=json.dumps(mounts) if mounts else None,
            build_memory=build_memory,
        )
        self._address = info["address"]
        self._container_id = info["container_id"]

    @property
    def address(self) -> str:
        return self._address

    @property
    def container_id(self) -> str:
        return self._container_id

    def shutdown(self) -> None:
        from .._rlmesh import sandbox_stop_env

        sandbox_stop_env(container_id=self._container_id)

    def __enter__(self) -> SandboxModel:
        return self

    def __exit__(self, *exc: object) -> bool:
        self.shutdown()
        return False
