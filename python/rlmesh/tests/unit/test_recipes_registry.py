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
    unregister,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    clear_registry()
    yield
    clear_registry()


def _gym_recipe(name: str = "safety/point-goal") -> Recipe:
    return Recipe(
        name=name,
        make=GymMake(env_id="SafetyPointGoal1-v0", kwargs={"render_mode": "rgb_array"}),
        build=Build(
            base="python:3.11-slim",
            pip=[PipInstall(packages=["safety-gymnasium==1.0.0"])],
        ),
        requires=Requires(imports=["safety_gymnasium"]),
    )


def test_register_and_resolve() -> None:
    recipe = register(_gym_recipe())
    assert resolve("safety/point-goal") == recipe
    assert registered_names() == ("safety/point-goal",)


def test_resolve_missing_raises_with_listing() -> None:
    register(_gym_recipe("a/one"))
    with pytest.raises(RecipeNotFoundError, match="a/one"):
        resolve("does/not-exist")


def test_register_rejects_shadowing() -> None:
    register(_gym_recipe())
    other = Recipe(name="safety/point-goal", make=GymMake(env_id="Different-v0"))
    with pytest.raises(ValueError, match="already registered"):
        register(other)


def test_register_idempotent_for_equal_recipe() -> None:
    register(_gym_recipe())
    # Re-registering an identical recipe is a no-op, not a shadow conflict.
    register(_gym_recipe())
    assert registered_names() == ("safety/point-goal",)


def test_register_overwrite() -> None:
    register(_gym_recipe())
    other = Recipe(name="safety/point-goal", make=GymMake(env_id="Different-v0"))
    register(other, overwrite=True)
    assert resolve("safety/point-goal").make == GymMake(env_id="Different-v0")


def test_unregister() -> None:
    register(_gym_recipe())
    unregister("safety/point-goal")
    assert registered_names() == ()
    unregister("safety/point-goal")  # absent is a no-op


def test_recipe_to_sandbox_args_flat_gym() -> None:
    args = recipe_to_sandbox_args(_gym_recipe())
    assert args.source == "SafetyPointGoal1-v0"
    assert args.packages == ("safety-gymnasium==1.0.0",)
    assert args.imports == ("safety_gymnasium",)
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
