from __future__ import annotations

import os
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
from rlmesh.recipes import (
    GymMake,
    HfMake,
    PyMake,
    Recipe,
    Setup,
    UnsupportedRecipeError,
    build,
)
from rlmesh.recipes._schema import FileWrite


def test_build_gym_recipe_constructs_real_env() -> None:
    env = build(Recipe(name="cart/pole", make=GymMake(env_id="CartPole-v1")))
    try:
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert isinstance(info, dict)
    finally:
        env.close()


def test_build_gym_recipe_forwards_kwargs() -> None:
    env = build(
        Recipe(
            name="cart/pole",
            make=GymMake(env_id="CartPole-v1", kwargs={"render_mode": "rgb_array"}),
        )
    )
    try:
        assert env.render_mode == "rgb_array"  # type: ignore[attr-defined]
    finally:
        env.close()


def test_build_rejects_build_only_base() -> None:
    from rlmesh.recipes import Build

    recipe = Recipe(name="base/img", build=Build(base="python:3.11-slim"))
    with pytest.raises(ValueError, match="build-only base"):
        build(recipe)


def test_build_hf_recipe_unsupported_in_process() -> None:
    recipe = Recipe(name="hf/env", make=HfMake(repo="org/env"))
    with pytest.raises(UnsupportedRecipeError, match="sandbox"):
        build(recipe)


def test_build_applies_setup_env() -> None:
    key = "RLMESH_TEST_SETUP_ENV"
    os.environ.pop(key, None)
    try:
        env = build(
            Recipe(
                name="cart/pole",
                make=GymMake(env_id="CartPole-v1"),
                setup=Setup(env={key: "applied"}),
            )
        )
        try:
            assert os.environ[key] == "applied"
        finally:
            env.close()
    finally:
        os.environ.pop(key, None)


def test_build_setup_files_unsupported_in_process() -> None:
    recipe = Recipe(
        name="cart/pole",
        make=GymMake(env_id="CartPole-v1"),
        setup=Setup(files=[FileWrite(path="x.txt", contents="hi")]),
    )
    # The message must not advise "run it in a sandbox" -- the sandbox bootstrap
    # calls this same apply_setup, so that advice would be impossible to follow.
    with pytest.raises(
        UnsupportedRecipeError, match=r"setup\.files is not applied yet"
    ) as excinfo:
        build(recipe)
    assert "sandbox" not in str(excinfo.value) or "local or sandbox" in str(
        excinfo.value
    )


@pytest.fixture
def py_factory_module(tmp_path: Path) -> Iterator[str]:
    """Write an importable factory module and put it on sys.path."""
    module_name = "rlmesh_test_factory_mod"
    (tmp_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            class _Env:
                def __init__(self, label):
                    self.label = label

                def reset(self, *, seed=None, options=None):
                    return self.label, {}

                def step(self, action):
                    return self.label, 0.0, False, False, {}

            def make_env(label="default"):
                return _Env(label)
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    try:
        yield module_name
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(module_name, None)


def test_build_py_recipe_wires_entrypoint(py_factory_module: str) -> None:
    recipe = Recipe(
        name="acme/factory",
        make=PyMake(
            entrypoint=f"{py_factory_module}:make_env", kwargs={"label": "wired"}
        ),
    )
    env = build(recipe)
    obs, info = env.reset()
    assert obs == "wired"
    assert info == {}


def test_build_py_recipe_accepts_bootstrap_vectorization_mode(
    py_factory_module: str,
) -> None:
    # The bootstrap (load_recipe_env) always passes vectorization_mode="sync" for a
    # single env; a num_envs==1 py build must accept it rather than crash.
    recipe = Recipe(
        name="acme/factory",
        make=PyMake(entrypoint=f"{py_factory_module}:make_env"),
    )
    env = build(recipe, num_envs=1, vectorization_mode="sync")
    obs, _ = env.reset()
    assert obs == "default"


def test_build_py_recipe_ignores_vectorization_mode_for_single_env(
    py_factory_module: str,
) -> None:
    # build()'s PyMake guard is `num_envs != 1`, so a single-env py recipe silently
    # ignores vectorization_mode (a single env is not vectorized). This is reachable
    # via rlmesh.make(py_recipe, vectorization_mode="async").
    recipe = Recipe(
        name="acme/factory",
        make=PyMake(entrypoint=f"{py_factory_module}:make_env"),
    )
    env = build(recipe, num_envs=1, vectorization_mode="async")
    obs, _ = env.reset()
    assert obs == "default"


def test_build_py_recipe_rejects_true_vector_request(py_factory_module: str) -> None:
    recipe = Recipe(
        name="acme/factory",
        make=PyMake(entrypoint=f"{py_factory_module}:make_env"),
    )
    with pytest.raises(TypeError, match="gym sources only"):
        build(recipe, num_envs=4)


def test_build_recipe_with_adapter_degrades_without_adapters(
    py_factory_module: str,
) -> None:
    # rlmesh.adapters does not exist in this branch yet; a recipe with an adapter
    # must still construct (publishing is skipped with a single warning).
    import importlib.util

    if importlib.util.find_spec("rlmesh.adapters") is not None:
        pytest.skip("rlmesh.adapters is now available; degradation path is moot")

    recipe = Recipe(
        name="acme/factory",
        make=PyMake(entrypoint=f"{py_factory_module}:make_env"),
        adapter={"observation": {"kind": "box"}},
    )
    with pytest.warns(RuntimeWarning, match="rlmesh.adapters is not available"):
        env = build(recipe)
    obs, _ = env.reset()
    assert obs == "default"


class _FakeGymModule:
    """A stand-in gym module whose ``make`` either fails to find the env or succeeds."""

    def __init__(self, name: str, *, succeeds: bool) -> None:
        self.__name__ = name
        self._succeeds = succeeds

    def make(self, env_id: str, **kwargs: object) -> object:
        if not self._succeeds:
            raise NameNotFound(env_id)
        return _FakeEnv(env_id)


class NameNotFound(Exception):  # noqa: N818 -- name must match gymnasium's exactly
    """Mirrors gymnasium's NameNotFound (matched by is_env_lookup_error by name)."""


class _FakeEnv:
    def __init__(self, env_id: str) -> None:
        self.env_id = env_id

    def reset(self, *, seed: object = None, options: object = None) -> object:
        return self.env_id, {}

    def step(self, action: object) -> object:
        return self.env_id, 0.0, False, False, {}


def test_build_gym_recipe_falls_back_to_legacy_gym(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # First module (gymnasium-like) does not have the env registered; the loop must
    # move on and resolve it from the second (legacy gym-like) module.
    first = _FakeGymModule("gymnasium", succeeds=False)
    second = _FakeGymModule("gym", succeeds=True)
    monkeypatch.setattr(
        "rlmesh.recipes._construct.import_gym_modules", lambda: [first, second]
    )
    recipe = Recipe(name="legacy/only", make=GymMake(env_id="LegacyOnly-v0"))
    env = build(recipe)
    obs, _ = env.reset()
    assert obs == "LegacyOnly-v0"


def test_build_gym_recipe_raises_aggregated_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakeGymModule("gymnasium", succeeds=False)
    second = _FakeGymModule("gym", succeeds=False)
    monkeypatch.setattr(
        "rlmesh.recipes._construct.import_gym_modules", lambda: [first, second]
    )
    recipe = Recipe(name="legacy/only", make=GymMake(env_id="Nowhere-v0"))
    with pytest.raises(RuntimeError, match="failed to create gym environment"):
        build(recipe)
