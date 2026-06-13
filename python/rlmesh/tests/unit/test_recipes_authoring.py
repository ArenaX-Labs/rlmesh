from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
import rlmesh
from rlmesh import recipes
from rlmesh.recipes import EnvRecipe, RecipeValidationError


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    recipes.clear_registry()
    yield
    recipes.clear_registry()


@pytest.fixture
def authored_module(tmp_path: Path) -> Iterator[str]:
    """An importable module defining EnvRecipe subclasses (the realistic case)."""
    module_name = "rlmesh_test_authored"
    (tmp_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            from __future__ import annotations

            import rlmesh
            from rlmesh.recipes import Build, PipInstall, ProjectInstall

            class Heavy(rlmesh.EnvRecipe):
                name = "acme/heavy"
                build = Build(
                    base="nvidia/cuda:12.4.1-runtime-ubuntu22.04",
                    pip=[PipInstall(["isaacsim"])],
                    project=ProjectInstall(src="."),
                    gpu=True,
                )

                def make(self, **kwargs):
                    import definitely_not_installed_xyz  # construct-time only
                    return definitely_not_installed_xyz.Env()

            class Cart(rlmesh.EnvRecipe):
                name = "cart/pole"

                def make(self, env_id="CartPole-v1", **kwargs):
                    import gymnasium
                    return gymnasium.make(env_id, **kwargs)

            class Lifecycle(rlmesh.EnvRecipe):
                name = "test/lifecycle"
                order: list[str] = []

                def prepare(self):
                    type(self).order.append("prepare")

                def make(self, **kwargs):
                    type(self).order.append("make")
                    return _StubEnv()

            class BadReturn(rlmesh.EnvRecipe):
                name = "test/bad-return"

                def make(self, **kwargs):
                    return {"not": "an env"}

            class NamelessChild(Cart):  # inherits Cart.name -- must be rejected
                pass

            class _StubEnv:
                def reset(self, *, seed=None, options=None):
                    return 0, {}
                def step(self, action):
                    return 0, 0.0, False, False, {}
                def close(self):
                    pass
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        yield module_name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(module_name, None)


def _module(name: str) -> object:
    import importlib

    return importlib.import_module(name)


# ----- projection is execution-free and import-safe -----


def test_to_recipe_projects_without_importing_env_deps(authored_module: str) -> None:
    heavy = _module(authored_module).Heavy  # type: ignore[attr-defined]
    # Heavy.make() imports a missing module; projection must NOT call it.
    recipe = heavy.to_recipe()
    assert recipe.name == "acme/heavy"
    assert recipe.make.entrypoint == f"{authored_module}:Heavy._rlmesh_construct"
    assert recipe.build.gpu is True
    assert recipe.build.project is not None


def test_check_is_dependency_free(authored_module: str) -> None:
    heavy = _module(authored_module).Heavy  # type: ignore[attr-defined]
    recipes.check(heavy.to_recipe())  # round-trip + entrypoint shape; imports nothing
    heavy.check()  # the classmethod shorthand


def test_to_recipe_rejects_local_class() -> None:
    class Local(EnvRecipe):
        name = "x/y"

        def make(self, **kwargs: object) -> object:
            return None

    with pytest.raises(RecipeValidationError, match="cannot import"):
        Local.to_recipe()


def test_to_recipe_requires_name(authored_module: str) -> None:
    class _NoName(EnvRecipe):
        def make(self, **kwargs: object) -> object:
            return None

    # _NoName is local, but the missing-name check fires first.
    with pytest.raises(RecipeValidationError, match="name"):
        _NoName.to_recipe()


# ----- the lifecycle: prepare then make -----


def test_entrypoint_runs_prepare_then_make(authored_module: str) -> None:
    from rlmesh._bootstrap.env import load_env_entrypoint

    mod = _module(authored_module)
    mod.Lifecycle.order.clear()  # type: ignore[attr-defined]
    env = load_env_entrypoint(f"{authored_module}:Lifecycle._rlmesh_construct")
    assert mod.Lifecycle.order == ["prepare", "make"]  # type: ignore[attr-defined]
    env.close()  # type: ignore[attr-defined]


def test_make_envrecipe_constructs_locally(authored_module: str) -> None:
    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    env = rlmesh.make(cart)
    try:
        assert env.reset(seed=0)[0] is not None
    finally:
        env.close()


def test_make_envrecipe_rejects_gym_sugar(authored_module: str) -> None:
    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="does not accept packages"):
        rlmesh.make(cart, packages=["x"])


# ----- registration + the provenance must-fix -----


def test_register_envrecipe_projects_and_resolves(authored_module: str) -> None:
    heavy = _module(authored_module).Heavy  # type: ignore[attr-defined]
    rlmesh.register(heavy)
    assert recipes.resolve("acme/heavy") == heavy.to_recipe()


def test_sandbox_source_for_envrecipe_with_project_is_installed(
    authored_module: str,
) -> None:
    """The headline fix: SandboxEnv(EnvRecipe-with-ProjectInstall) must NOT be Remote.

    A Remote recipe rejects ProjectInstall; stamping an in-process/EnvRecipe source
    as Installed is what lets the robotics-on-a-Mac one-liner launch.
    """
    from rlmesh.sandbox import _resolve_recipe_source

    heavy = _module(authored_module).Heavy  # type: ignore[attr-defined]
    display, recipe_json, provenance, context_root = _resolve_recipe_source(heavy)
    assert display == "acme/heavy"
    assert provenance == "installed"
    assert recipe_json is not None
    assert context_root is not None  # ProjectInstall present -> cwd staged


def test_sandbox_source_for_literal_recipe_is_installed() -> None:
    from rlmesh.recipes import GymMake, Recipe
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(name="acme/literal", make=GymMake(env_id="CartPole-v1"))
    _, _, provenance, _ = _resolve_recipe_source(recipe)
    assert provenance == "installed"


# ----- EnvServer(EnvRecipe) -----


def test_envserver_builds_envrecipe(authored_module: str) -> None:
    from rlmesh.server import _coerce_env

    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    env = _coerce_env(cart)
    try:
        assert hasattr(env, "reset") and hasattr(env, "step")
    finally:
        getattr(env, "close", lambda: None)()


# ----- footgun guards (from the adversarial review) -----


def test_make_rejects_envrecipe_instance(authored_module: str) -> None:
    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="not an instance"):
        rlmesh.make(cart())  # the class is correct; an instance is the mistake


def test_envserver_rejects_envrecipe_instance(authored_module: str) -> None:
    from rlmesh.server import _coerce_env

    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="not an instance"):
        _coerce_env(cart())


def test_make_envrecipe_rejects_vectorization(authored_module: str) -> None:
    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="num_envs"):
        rlmesh.make(cart, num_envs=4)
    with pytest.raises(TypeError, match="num_envs"):
        rlmesh.make(cart, vectorization_mode="async")


def test_make_py_recipe_rejects_vectorization() -> None:
    from rlmesh.recipes import PyMake, Recipe, build

    recipe = Recipe(name="a/py", make=PyMake(entrypoint="mod:make"))
    with pytest.raises(TypeError, match="num_envs"):
        build(recipe, num_envs=4)


def test_construct_authored_rejects_non_env_return(authored_module: str) -> None:
    bad = _module(authored_module).BadReturn  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="did not return an environment"):
        rlmesh.make(bad)


def test_nameless_subclass_of_named_is_rejected(authored_module: str) -> None:
    child = _module(authored_module).NamelessChild  # type: ignore[attr-defined]
    with pytest.raises(RecipeValidationError, match="declare its own"):
        child.to_recipe()
