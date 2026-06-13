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
        assert env.render_mode == "rgb_array"
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
    from rlmesh.sandbox import _resolve_recipe_source

    recipes.register(Recipe(name="acme/env", make=GymMake(env_id="CartPole-v1")))
    display, recipe_json, provenance, context_root = _resolve_recipe_source("acme/env")
    assert display == "acme/env"
    assert recipe_json is not None and provenance == "installed"
    assert context_root is None


def test_resolve_recipe_source_for_literal_recipe() -> None:
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(name="acme/env", make=GymMake(env_id="CartPole-v1"))
    display, recipe_json, provenance, _ = _resolve_recipe_source(recipe)
    assert display == "acme/env"
    assert recipe_json == recipe.to_json()
    assert provenance == "remote"


def test_resolve_recipe_source_passes_through_gym_id() -> None:
    from rlmesh.sandbox import _resolve_recipe_source

    assert _resolve_recipe_source("CartPole-v1") == ("CartPole-v1", None, None, None)


def test_resolve_recipe_source_defaults_context_root_for_project() -> None:
    from rlmesh.sandbox import _resolve_recipe_source

    recipe = Recipe(
        name="acme/env",
        make=GymMake(env_id="CartPole-v1"),
        build=Build(project=ProjectInstall(src=".", dest="/opt/acme")),
    )
    _, _, _, context_root = _resolve_recipe_source(recipe)
    assert context_root is not None


def test_start_sandbox_forwards_recipe_json_for_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import sandbox

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
    from rlmesh import sandbox

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


def test_make_safety_gym_one_liner_shape() -> None:
    # The locked acceptance case: a gym recipe with flat packages round-trips
    # through make() coercion without needing a sandbox.
    recipe = Recipe(
        name="safety/point-goal",
        make=GymMake(env_id="SafetyPointGoal1-v0"),
        build=Build(pip=[PipInstall(packages=["safety-gymnasium==1.0.0"])]),
        requires=Requires(imports=["safety_gymnasium"]),
    )
    recipes.register(recipe)
    assert recipes.resolve("safety/point-goal") == recipe
