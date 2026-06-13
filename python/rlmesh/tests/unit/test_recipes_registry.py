from __future__ import annotations

from collections.abc import Iterator

import pytest
from rlmesh.recipes import (
    Build,
    Fetch,
    GymMake,
    HfMake,
    PipInstall,
    ProjectInstall,
    PyMake,
    Recipe,
    RecipeNotFoundError,
    Requires,
    Setup,
    UnsupportedRecipeError,
    clear_registry,
    recipe_to_sandbox_args,
    register,
    registered_names,
    resolve,
    resolve_from_recipe,
    unregister,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    clear_registry()
    yield
    clear_registry()


def _gym_recipe(name: str = "atari/breakout") -> Recipe:
    return Recipe(
        name=name,
        make=GymMake(env_id="ALE/Breakout-v5", kwargs={"render_mode": "rgb_array"}),
        build=Build(
            base="python:3.11-slim",
            pip=[PipInstall(packages=["ale-py"])],
        ),
        requires=Requires(imports=["ale_py"]),
    )


def test_register_and_resolve() -> None:
    recipe = register(_gym_recipe())
    assert resolve("atari/breakout") == recipe
    assert registered_names() == ("atari/breakout",)


def test_resolve_missing_raises_with_listing() -> None:
    register(_gym_recipe("a/one"))
    with pytest.raises(RecipeNotFoundError, match="a/one"):
        resolve("does/not-exist")


def test_register_rejects_shadowing() -> None:
    register(_gym_recipe())
    other = Recipe(name="atari/breakout", make=GymMake(env_id="Different-v0"))
    with pytest.raises(ValueError, match="already registered"):
        register(other)


def test_register_idempotent_for_equal_recipe() -> None:
    register(_gym_recipe())
    # Re-registering an identical recipe is a no-op, not a shadow conflict.
    register(_gym_recipe())
    assert registered_names() == ("atari/breakout",)


def test_register_overwrite() -> None:
    register(_gym_recipe())
    other = Recipe(name="atari/breakout", make=GymMake(env_id="Different-v0"))
    register(other, overwrite=True)
    assert resolve("atari/breakout").make == GymMake(env_id="Different-v0")


def test_unregister() -> None:
    register(_gym_recipe())
    unregister("atari/breakout")
    assert registered_names() == ()
    unregister("atari/breakout")  # absent is a no-op


def test_recipe_to_sandbox_args_flat_gym() -> None:
    args = recipe_to_sandbox_args(_gym_recipe())
    assert args.source == "ALE/Breakout-v5"
    assert args.packages == ("ale-py",)
    assert args.imports == ("ale_py",)
    assert args.base_image == "python:3.11-slim"
    assert args.kwargs == {"render_mode": "rgb_array"}


def test_recipe_to_sandbox_args_flattens_multiple_pip_steps() -> None:
    recipe = Recipe(
        name="a/multi",
        make=GymMake(env_id="E-v0"),
        build=Build(pip=[PipInstall(packages=["a", "b"]), PipInstall(packages=["c"])]),
    )
    assert recipe_to_sandbox_args(recipe).packages == ("a", "b", "c")


def test_recipe_to_sandbox_args_rejects_non_gym() -> None:
    recipe = Recipe(name="a/py", make=HfMake(repo="org/env"))
    with pytest.raises(UnsupportedRecipeError, match="non-gym"):
        recipe_to_sandbox_args(recipe)


def test_recipe_to_sandbox_args_rejects_structured_build() -> None:
    recipe = Recipe(
        name="a/heavy",
        make=GymMake(env_id="E-v0"),
        build=Build(
            system=["cmake"],
            fetch=[Fetch(kind="git", repo="https://x/r.git", ref="a" * 40)],
            project=ProjectInstall(),
            gpu=True,
        ),
    )
    with pytest.raises(UnsupportedRecipeError, match="build deriver"):
        recipe_to_sandbox_args(recipe)


def test_recipe_to_sandbox_args_rejects_indexed_pip() -> None:
    recipe = Recipe(
        name="a/indexed",
        make=GymMake(env_id="E-v0"),
        build=Build(
            pip=[
                PipInstall(
                    packages=["torch"], index_url="https://download.pytorch.org/whl/cpu"
                )
            ]
        ),
    )
    with pytest.raises(UnsupportedRecipeError, match="index URL"):
        recipe_to_sandbox_args(recipe)


def test_recipe_to_sandbox_args_rejects_setup() -> None:
    recipe = Recipe(
        name="a/setup",
        make=GymMake(env_id="E-v0"),
        setup=Setup(env={"LIBERO_TASK": "x"}),
    )
    with pytest.raises(UnsupportedRecipeError, match="setup"):
        recipe_to_sandbox_args(recipe)


# ----- from_recipe build reuse -----


def _base_build() -> Build:
    return Build(
        base="nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        system=["cmake"],
        pip=[PipInstall(packages=["torch==2.0.0"])],
        gpu=True,
    )


def test_resolve_from_recipe_inlines_base_build() -> None:
    register(Recipe(name="droid/base", build=_base_build()))
    child = Recipe(
        name="droid/scene1",
        make=PyMake(entrypoint="robot_env:make"),
        build=Build(from_recipe="droid/base"),
        setup=Setup(env={"SCENE": "1"}),
    )
    resolved = resolve_from_recipe(child)
    assert resolved.build == _base_build()
    # make/setup are preserved; only the build is inlined.
    assert resolved.make == child.make
    assert resolved.setup == child.setup


def test_from_recipe_family_shares_identical_build() -> None:
    register(Recipe(name="droid/base", build=_base_build()))
    scenes = [
        resolve_from_recipe(
            Recipe(
                name=f"droid/scene{i}",
                make=PyMake(entrypoint=f"robot_env:scene{i}"),
                build=Build(from_recipe="droid/base"),
            )
        )
        for i in range(1, 4)
    ]
    # Every task in the family shares one build (and thus one image/build_hash).
    assert scenes[0].build == scenes[1].build == scenes[2].build == _base_build()
    # ...while their factories differ.
    assert len({repr(s.make) for s in scenes}) == 3


def test_recipe_without_from_recipe_is_unchanged() -> None:
    recipe = Recipe(name="a/plain", make=GymMake(env_id="E-v0"), build=Build(gpu=True))
    assert resolve_from_recipe(recipe) is recipe


def test_from_recipe_rejects_extra_build_fields() -> None:
    register(Recipe(name="droid/base", build=_base_build()))
    child = Recipe(
        name="droid/scene1",
        make=PyMake(entrypoint="robot_env:make"),
        build=Build(from_recipe="droid/base", gpu=True),
    )
    with pytest.raises(ValueError, match="exclusive with other build fields"):
        resolve_from_recipe(child)


def test_from_recipe_missing_base_raises() -> None:
    child = Recipe(
        name="droid/scene1",
        make=PyMake(entrypoint="robot_env:make"),
        build=Build(from_recipe="droid/missing"),
    )
    with pytest.raises(RecipeNotFoundError):
        resolve_from_recipe(child)


def test_from_recipe_detects_cycles() -> None:
    register(Recipe(name="a/one", build=Build(from_recipe="a/two")))
    register(Recipe(name="a/two", build=Build(from_recipe="a/one")))
    with pytest.raises(ValueError, match="cycle"):
        resolve_from_recipe(resolve("a/one"))
