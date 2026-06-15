from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest
import rlmesh
from rlmesh import recipes
from rlmesh._bootstrap.env import load_env_from_spec
from rlmesh.recipes import Build, GymMake, PipInstall, ProjectInstall, Recipe, Requires


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    recipes.clear_registry()
    yield
    recipes.clear_registry()


# ----- rlmesh.make -----


def test_make_gym_id_fallthrough() -> None:
    env = rlmesh.make("CartPole-v1")
    try:
        obs, _ = env.reset(seed=0)
        assert obs is not None
    finally:
        env.close()


def test_make_forwards_gym_kwargs() -> None:
    env = rlmesh.make("CartPole-v1", render_mode="rgb_array")
    try:
        assert env.render_mode == "rgb_array"  # type: ignore[attr-defined]
    finally:
        env.close()


def test_make_registered_recipe() -> None:
    recipes.register(Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1")))
    env = rlmesh.make("cart/pole")
    try:
        assert env.reset()[0] is not None
    finally:
        env.close()


def test_make_literal_recipe() -> None:
    env = rlmesh.make(Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1")))
    try:
        assert env.reset()[0] is not None
    finally:
        env.close()


def test_make_recipe_rejects_extra_kwargs() -> None:
    with pytest.raises(TypeError, match="bake them"):
        rlmesh.make(Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1")), x=1)


# ----- bootstrap dispatch (kind="recipe") -----


def test_bootstrap_recipe_dispatch_builds_env() -> None:
    spec = {
        "kind": "recipe",
        "document": {
            "name": "cart/pole",
            "make": {"kind": "gym", "env_id": "CartPole-v1"},
        },
        "num_envs": 1,
        "vectorization_mode": "sync",
    }
    env = cast("rlmesh.RemoteEnv", load_env_from_spec(spec))
    try:
        assert env.reset()[0] is not None  # type: ignore[attr-defined]
    finally:
        env.close()  # type: ignore[attr-defined]


# ----- SandboxEnv source resolution -----


def test_resolve_recipe_source_for_registered_name() -> None:
    from rlmesh.sandbox._export import resolve_recipe_source

    recipes.register(Recipe(name="acme/env", make=GymMake(env_id="CartPole-v1")))
    display, recipe_json, provenance, context_root, _ = resolve_recipe_source(
        "acme/env"
    )
    assert display == "acme/env"
    assert recipe_json is not None and provenance == "installed"
    assert context_root is None


def test_resolve_recipe_source_for_literal_recipe() -> None:
    from rlmesh.sandbox._export import resolve_recipe_source

    recipe = Recipe(name="acme/env", make=GymMake(env_id="CartPole-v1"))
    display, recipe_json, provenance, _, _ = resolve_recipe_source(recipe)
    assert display == "acme/env"
    assert recipe_json == recipe.to_json()
    # An in-process literal Recipe is Installed (it came from your code); Remote is
    # reserved for an untrusted external document.
    assert provenance == "installed"


def test_resolve_recipe_source_passes_through_gym_id() -> None:
    from rlmesh.sandbox._export import resolve_recipe_source

    assert resolve_recipe_source("CartPole-v1") == (
        "CartPole-v1",
        None,
        None,
        None,
        (),
    )


def test_resolve_recipe_source_defaults_context_root_for_project() -> None:
    from rlmesh.sandbox._export import resolve_recipe_source

    recipe = Recipe(
        name="acme/env",
        make=GymMake(env_id="CartPole-v1"),
        build=Build(project=ProjectInstall(src=".", dest="/opt/acme")),
    )
    _, _, _, context_root, _ = resolve_recipe_source(recipe)
    assert context_root is not None


def test_start_sandbox_forwards_recipe_json_for_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh.sandbox import session as sandbox

    captured: dict[str, object] = {}

    def fake_start(source: str, **kwargs: object) -> dict[str, str]:
        captured["source"] = source
        captured.update(kwargs)
        return {
            "requested_source": source,
            "resolved_source": source,
            "address": "tcp://127.0.0.1:1",
            "container_id": "c1",
        }

    monkeypatch.setattr(sandbox, "_sandbox_start_env", fake_start)
    recipes.register(Recipe(name="acme/env", make=GymMake(env_id="CartPole-v1")))
    sandbox._start_sandbox(
        "acme/env",
        base_image=None,
        rlmesh_package=None,
        packages=None,
        imports=None,
        trust_remote_code=False,
        allow_unpinned_hf=False,
        num_envs=1,
        vectorization_mode=None,
        gym_make_kwargs={},
    )
    assert captured["recipe_provenance"] == "installed"
    assert "recipe_json" in captured


def test_start_sandbox_omits_recipe_args_for_gym(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh.sandbox import session as sandbox

    captured: dict[str, object] = {}

    def fake_start(source: str, **kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {
            "requested_source": source,
            "resolved_source": source,
            "address": "tcp://127.0.0.1:1",
            "container_id": "c1",
        }

    monkeypatch.setattr(sandbox, "_sandbox_start_env", fake_start)
    sandbox._start_sandbox(
        "CartPole-v1",
        base_image=None,
        rlmesh_package=None,
        packages=None,
        imports=None,
        trust_remote_code=False,
        allow_unpinned_hf=False,
        num_envs=1,
        vectorization_mode=None,
        gym_make_kwargs={},
    )
    assert "recipe_json" not in captured
    assert "recipe_provenance" not in captured


# ----- EnvServer(recipe) -----


def test_envserver_coerces_recipe() -> None:
    from rlmesh.server import _coerce_env

    env = _coerce_env(Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1")))
    try:
        assert hasattr(env, "reset") and hasattr(env, "step")
    finally:
        getattr(env, "close", lambda: None)()


def test_make_gym_one_liner_shape() -> None:
    # A gym recipe (Atari, registered into Gymnasium by importing ale_py) with flat
    # packages round-trips through make() coercion without needing a sandbox.
    recipe = Recipe(
        name="atari/breakout",
        make=GymMake(env_id="ALE/Breakout-v5"),
        build=Build(pip=[PipInstall(packages=["ale-py"])]),
        requires=Requires(imports=["ale_py"]),
    )
    recipes.register(recipe)
    assert recipes.resolve("atari/breakout") == recipe


# ----- flat register / make sugar (Part E) -----


def test_register_flat_gym_sugar() -> None:
    recipe = rlmesh.register(
        "atari/breakout", gym="ALE/Breakout-v5", packages=["ale-py"], imports=["ale_py"]
    )
    assert recipe.make == GymMake(env_id="ALE/Breakout-v5")
    assert recipe.build.pip == (PipInstall(packages=["ale-py"]),)
    assert recipe.requires.imports == ("ale_py",)
    assert recipes.resolve("atari/breakout") == recipe


def test_register_flat_factory_sugar() -> None:
    from rlmesh.recipes import PyMake

    recipe = rlmesh.register(
        "safety/pg", factory="safety_env:make", packages=["safety-gymnasium==1.0.0"]
    )
    assert recipe.make == PyMake(entrypoint="safety_env:make")
    assert recipe.build.pip == (PipInstall(packages=["safety-gymnasium==1.0.0"]),)


def test_register_needs_exactly_one_of_gym_or_factory() -> None:
    with pytest.raises(TypeError, match="exactly one of gym= or factory="):
        rlmesh.register("x/y")
    with pytest.raises(TypeError, match="exactly one of gym= or factory="):
        rlmesh.register("x/y", gym="E-v0", factory="m:f")


def test_register_object_form_rejects_sugar() -> None:
    recipe = Recipe(name="a/b", make=GymMake("E-v0"))
    with pytest.raises(TypeError, match="takes no gym"):
        rlmesh.register(recipe, gym="E-v0")  # type: ignore[call-overload]


def test_register_factory_rejects_imports() -> None:
    with pytest.raises(TypeError, match="gym-only"):
        rlmesh.register("x/y", factory="m:f", imports=["x"])


def test_register_gym_rejects_pip_shaped_imports() -> None:
    with pytest.raises(TypeError, match="looks like a package"):
        rlmesh.register("x/y", gym="E-v0", imports=["ale-py==1.0"])
    with pytest.raises(TypeError, match="looks like a package"):
        rlmesh.register("x/y", gym="E-v0", packages=["ale-py"], imports=["ale-py"])


def test_make_gym_id_forwards_imports() -> None:
    # imports= registers the env on import; CartPole needs none, but the path runs.
    env = rlmesh.make("CartPole-v1", imports=[])
    try:
        assert env.reset(seed=0)[0] is not None
    finally:
        env.close()


def test_register_gym_sugar_accepts_colon_id() -> None:
    recipe = rlmesh.register("sai/squid", gym="sai_pygame:SquidHunt-v0")
    assert recipe.make == GymMake(env_id="sai_pygame:SquidHunt-v0")
