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
        assert env.render_mode == "rgb_array"
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
    with pytest.raises(UnsupportedRecipeError, match=r"setup\.files"):
        build(recipe)


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
