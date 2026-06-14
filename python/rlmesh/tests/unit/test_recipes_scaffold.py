from __future__ import annotations

import textwrap
from collections.abc import Iterator

import pytest
from rlmesh import recipes
from rlmesh.recipes import (
    Build,
    GymMake,
    PipInstall,
    ProjectInstall,
    PyMake,
    Recipe,
)
from rlmesh.recipes.scaffold import scaffold_from_pyproject, scaffold_recipe


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    recipes.clear_registry()
    yield
    recipes.clear_registry()


def _exec_recipe(source: str) -> Recipe:
    """Execute generated recipe source and return the registered RECIPE."""
    namespace: dict[str, object] = {}
    exec(compile(source, "<scaffold>", "exec"), namespace)
    recipe = namespace["RECIPE"]
    assert isinstance(recipe, Recipe)
    return recipe


def test_scaffold_generates_executable_recipe() -> None:
    result = scaffold_recipe(
        "acme/robot",
        "robot_env.factory:make",
        dependencies=["numpy>=1.26", "robosuite==1.4.1"],
    )
    recipe = _exec_recipe(result.recipe_source)
    assert recipe.name == "acme/robot"
    assert recipe.make == PyMake(entrypoint="robot_env.factory:make")
    # Plain deps collapse into one PyPI step.
    assert recipe.build.pip == (
        PipInstall(packages=["numpy>=1.26", "robosuite==1.4.1"]),
    )
    # Registering it (the generated source calls register) makes it resolvable.
    assert recipes.resolve("acme/robot") == recipe


def test_scaffold_indexed_dependency_gets_own_pip_step() -> None:
    result = scaffold_recipe(
        "acme/robot",
        "robot_env:make",
        dependencies=["torch==2.4.0", "numpy>=1.26"],
        uv_sources={"torch": {"index": "pytorch-cu124"}},
        uv_index=[
            {"name": "pytorch-cu124", "url": "https://download.pytorch.org/whl/cu124"}
        ],
    )
    recipe = _exec_recipe(result.recipe_source)
    pip = recipe.build.pip
    torch_step = next(s for s in pip if "torch==2.4.0" in s.packages)
    assert torch_step.index_url == "https://download.pytorch.org/whl/cu124"
    plain_step = next(s for s in pip if "numpy>=1.26" in s.packages)
    assert plain_step.index_url is None


def test_scaffold_guesses_gpu_from_markers() -> None:
    result = scaffold_recipe(
        "acme/isaac",
        "robot_env:make",
        dependencies=["isaacsim", "isaaclab[all]==2.2.0"],
    )
    recipe = _exec_recipe(result.recipe_source)
    assert recipe.build.gpu is True
    assert recipe.build.base is not None and "cuda" in recipe.build.base
    assert any("SimulationApp" in todo for todo in result.todos)
    assert "SimulationApp" in result.factory_source


def test_scaffold_no_gpu_for_plain_deps() -> None:
    result = scaffold_recipe("a/b", "m:f", dependencies=["gymnasium==1.0.0"])
    recipe = _exec_recipe(result.recipe_source)
    assert recipe.build.gpu is False
    assert recipe.build.base is None


def test_scaffold_detect_assets_emits_project_install() -> None:
    result = scaffold_recipe(
        "acme/robot",
        "robot_env:make",
        dependencies=["numpy"],
        detect_assets=True,
    )
    recipe = _exec_recipe(result.recipe_source)
    assert recipe.build.project == ProjectInstall(src=".", include=("assets/**",))


def test_scaffold_always_emits_system_runtime_todo() -> None:
    result = scaffold_recipe("a/b", "m:f", dependencies=["numpy"])
    assert any("system_runtime" in todo for todo in result.todos)


def test_scaffold_factory_stub_uses_entrypoint_callable() -> None:
    result = scaffold_recipe("a/b", "robot_env.factory:build_env", dependencies=[])
    assert "def build_env(" in result.factory_source


def test_scaffold_from_pyproject_parses_deps_and_index() -> None:
    text = textwrap.dedent(
        """
        [project]
        name = "robot-env"
        dependencies = ["torch==2.4.0", "numpy>=1.26"]

        [tool.uv.sources]
        torch = { index = "pytorch-cu124" }

        [[tool.uv.index]]
        name = "pytorch-cu124"
        url = "https://download.pytorch.org/whl/cu124"
        explicit = true
        """
    )
    result = scaffold_from_pyproject("acme/robot", "robot_env:make", text)
    recipe = _exec_recipe(result.recipe_source)
    torch_step = next(s for s in recipe.build.pip if "torch==2.4.0" in s.packages)
    assert torch_step.index_url == "https://download.pytorch.org/whl/cu124"


def test_scaffolded_recipe_is_buildable_shape() -> None:
    # The generated recipe is a normal Recipe; it round-trips and validates like
    # any hand-written one.
    result = scaffold_recipe("a/b", "m:f", dependencies=["numpy"])
    recipe = _exec_recipe(result.recipe_source)
    assert Recipe.from_json(recipe.to_json()) == recipe
    # And a hand-written equivalent without the scaffolder still works.
    assert Recipe(name="a/b", make=GymMake(env_id="E-v0"), build=Build())
