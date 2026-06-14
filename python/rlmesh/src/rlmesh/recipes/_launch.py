"""Translate a flat gym recipe into today's ``SandboxEnv`` launch arguments.

Slice 1 of the recipe system reuses the existing sandbox path: a ``GymMake``
recipe whose ``build`` is expressible as a base image plus flat ``pip install``
lines maps directly onto ``SandboxEnv(source, packages=, imports=, **kwargs)``.
A recipe that needs the structured build deriver (apt, git fetches, a project
install, per-step pip indices, GPU, ...) is rejected here with a pointer to that
path, which lands in a later slice -- so the flat path never silently drops a
build instruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._schema import Build, GymMake, PipInstall, Recipe, Setup

__all__ = [
    "SandboxLaunchArgs",
    "UnsupportedRecipeError",
    "recipe_to_sandbox_args",
]


class UnsupportedRecipeError(NotImplementedError):
    """Raised when a recipe needs the structured build deriver, not the flat path.

    The flat path only covers ``GymMake`` recipes whose build is a base image plus
    simple ``pip install`` lines. Anything else (apt packages, git fetches, a
    project install, per-step pip indices, GPU, raw commands, a verbatim
    Dockerfile, or construct-time setup) requires the recipe -> Dockerfile deriver.
    """


@dataclass(frozen=True)
class SandboxLaunchArgs:
    """The arguments needed to launch a flat gym recipe via ``SandboxEnv``."""

    source: str
    packages: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()
    base_image: str | None = None
    kwargs: dict[str, object] = field(default_factory=dict)


def _flat_pip_packages(pip: tuple[PipInstall, ...]) -> tuple[str, ...]:
    """Flatten simple pip steps to a package list, or reject index/option steps."""
    packages: list[str] = []
    for step in pip:
        if (
            step.index_url is not None
            or step.extra_index_urls
            or step.no_deps
            or step.pre
            or step.requirements is not None
        ):
            raise UnsupportedRecipeError(
                "PipInstall with an index URL / no_deps / pre / -r requirements needs "
                "the build deriver; the flat path supports plain package lists only"
            )
        packages.extend(step.packages)
    return tuple(packages)


def _reject_non_flat_build(build: Build) -> None:
    """Reject every build field the flat sandbox path cannot express."""
    non_flat: list[str] = []
    if build.from_recipe is not None:
        non_flat.append("from_recipe")
    if build.system:
        non_flat.append("system")
    if build.system_runtime:
        non_flat.append("system_runtime")
    if build.project is not None:
        non_flat.append("project")
    if build.fetch:
        non_flat.append("fetch")
    if build.env:
        non_flat.append("env")
    if build.pythonpath:
        non_flat.append("pythonpath")
    if build.gpu:
        non_flat.append("gpu")
    if build.installer != "pip":
        non_flat.append("installer")
    if build.run_as is not None:
        non_flat.append("run_as")
    if build.commands:
        non_flat.append("commands")
    if build.dockerfile is not None:
        non_flat.append("dockerfile")
    if non_flat:
        raise UnsupportedRecipeError(
            f"build fields {sorted(non_flat)} need the build deriver; the flat "
            "sandbox path supports only base= and plain pip packages"
        )


def recipe_to_sandbox_args(recipe: Recipe) -> SandboxLaunchArgs:
    """Translate a flat ``GymMake`` recipe into ``SandboxEnv`` launch arguments.

    Raises:
        UnsupportedRecipeError: If the recipe needs the structured build deriver
            (non-gym factory, structured build steps, or construct-time setup).
    """
    if not isinstance(recipe.make, GymMake):
        raise UnsupportedRecipeError(
            f"recipe {recipe.name!r} has a non-gym make; only GymMake is supported "
            "by the flat sandbox path in this slice"
        )
    if recipe.setup != Setup():
        raise UnsupportedRecipeError(
            f"recipe {recipe.name!r} declares construct-time setup, which the flat "
            "sandbox path does not apply; use the build/sandbox deriver"
        )
    _reject_non_flat_build(recipe.build)
    return SandboxLaunchArgs(
        source=recipe.make.env_id,
        packages=_flat_pip_packages(tuple(recipe.build.pip)),
        imports=tuple(recipe.requires.imports),
        base_image=recipe.build.base,
        kwargs=dict(recipe.make.kwargs),
    )
