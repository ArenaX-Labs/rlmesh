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

            import gymnasium as _gym
            import numpy as _np
            import rlmesh
            from rlmesh.adapters import (
                ACTION_GRIPPER, IMAGE_PRIMARY, ActionComponent, ActionLayout,
                EnvTags, ImageTag,
            )
            from rlmesh.recipes import Build, PipInstall, ProjectInstall

            _TAGS = EnvTags(
                observation={"img": ImageTag(role=IMAGE_PRIMARY)},
                action=ActionLayout(ActionComponent(ACTION_GRIPPER, dim=2)),
            )

            class _TaggedStub:
                observation_space = _gym.spaces.Dict({"img": _gym.spaces.Box(0, 255, (8, 8, 3), _np.uint8)})
                action_space = _gym.spaces.Box(-1, 1, (2,), _np.float32)
                metadata: dict = {}
                def reset(self, *, seed=None, options=None):
                    return {"img": _np.zeros((8, 8, 3), _np.uint8)}, {}
                def step(self, action):
                    return {"img": _np.zeros((8, 8, 3), _np.uint8)}, 0.0, False, False, {}

            class Tagged(rlmesh.EnvRecipe):
                name = "test/tagged"
                tags = _TAGS
                def make(self, **kwargs):
                    return _TaggedStub()

            class TaggedConflict(rlmesh.EnvRecipe):
                name = "test/tagged-conflict"
                tags = _TAGS
                def make(self, **kwargs):
                    return rlmesh.adapters.tag(_TaggedStub(), _TAGS)

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

            @rlmesh.register
            class Registered(rlmesh.EnvRecipe):  # auto-registers at import
                name = "test/registered"

                def make(self, **kwargs):
                    return _StubEnv()

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


def test_class_tags_project_onto_recipe_adapter(authored_module: str) -> None:
    module = _module(authored_module)
    tagged = module.Tagged  # type: ignore[attr-defined]
    recipe = tagged.to_recipe()
    # The env-side mirror of ModelRecipe.spec: declared tags ride recipe.adapter.
    assert recipe.adapter is module._TAGS  # type: ignore[attr-defined]
    assert recipe.kind == "env"
    # A tag-less recipe leaves adapter empty (byte-stable for existing recipes).
    assert module.Cart.to_recipe().adapter is None  # type: ignore[attr-defined]


def test_class_tags_published_on_constructed_env(authored_module: str) -> None:
    from rlmesh.adapters import EnvTags
    from rlmesh.recipes._authoring import construct_authored

    tagged = _module(authored_module).Tagged  # type: ignore[attr-defined]
    env = construct_authored(tagged)
    # The framework attached the declared tags, so the served env publishes them.
    assert EnvTags.from_metadata(env.metadata) is not None


def test_class_tags_and_make_tags_conflict_fails_loud(authored_module: str) -> None:
    from rlmesh.recipes._authoring import construct_authored

    conflict = _module(authored_module).TaggedConflict  # type: ignore[attr-defined]
    with pytest.raises(RecipeValidationError, match="one place"):
        construct_authored(conflict)


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


def test_required_init_arg_raises_recipe_aware_error() -> None:
    # cls() is called with no args during construction; a required-arg __init__ would
    # otherwise fail with a confusing native TypeError.
    from rlmesh.recipes._authoring import construct_authored

    class NeedsArg(EnvRecipe):
        name = "x/needs-arg"

        def __init__(self, handle: object) -> None:
            self._handle = handle

        def make(self, **kwargs: object) -> object:
            return self._handle

    with pytest.raises(TypeError, match="make\\(self, \\*\\*kwargs\\)"):
        construct_authored(NeedsArg)


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


# ----- the @register decorator -----


def test_register_decorator_binds_class_and_registers(authored_module: str) -> None:
    registered = _module(authored_module).Registered  # type: ignore[attr-defined]
    # @register returned the class (not a Recipe), so the name stays usable...
    assert isinstance(registered, type)
    assert issubclass(registered, EnvRecipe)
    # ...and the projected recipe was stored under its name.
    assert recipes.resolve("test/registered") == registered.to_recipe()


def test_register_decorator_call_form_returns_class(authored_module: str) -> None:
    cart = _module(authored_module).Cart  # type: ignore[attr-defined]
    assert rlmesh.register(cart) is cart  # register(cls) is decorator-equivalent
    assert recipes.resolve("cart/pole") == cart.to_recipe()


def test_register_decorator_rejects_unimportable() -> None:
    # A recipe the container cannot import (here a local class; __main__ is the
    # other case) cannot travel by reference, so @register fails fast rather than
    # storing an unbuildable entry.
    class Local(EnvRecipe):
        name = "x/local"

        def make(self, **kwargs: object) -> object:
            return None

    with pytest.raises(RecipeValidationError, match="cannot import"):
        rlmesh.register(Local)
