from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace

import pytest


def test_load_environment_imports_registration_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.env import load_environment

    imported: list[str] = []
    registration_module = ModuleType("fake_registration")

    gymnasium = ModuleType("gymnasium")

    def make(env_id: str, **kwargs: object) -> tuple[str, str, dict[str, object], bool]:
        return env_id, "gymnasium", kwargs, "fake_registration" in imported

    gymnasium.make = make  # type: ignore[attr-defined]

    real_import = importlib.import_module

    def import_module(name: str) -> ModuleType:
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setitem(sys.modules, "fake_registration", registration_module)
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    env = load_environment(
        "CartPole-v1",
        ["fake_registration"],
        num_envs=1,
        kwargs={"render_mode": "rgb_array"},
    )

    assert env == (
        "CartPole-v1",
        "gymnasium",
        {"render_mode": "rgb_array"},
        True,
    )


def test_load_environment_falls_back_to_gym_for_missing_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.env import load_environment

    name_not_found = type("NameNotFound", (Exception,), {})

    gymnasium = ModuleType("gymnasium")

    def missing_make(env_id: str, **kwargs: object) -> object:
        raise name_not_found(env_id)

    gymnasium.make = missing_make  # type: ignore[attr-defined]

    gym = ModuleType("gym")

    def gym_make(env_id: str, **kwargs: object) -> tuple[str, str, dict[str, object]]:
        return "gym", env_id, kwargs

    gym.make = gym_make  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gym", gym)

    assert load_environment("OnlyInGym-v0", [], num_envs=1) == (
        "gym",
        "OnlyInGym-v0",
        {},
    )


def test_make_gym_environment_prefers_make_vec() -> None:
    from rlmesh._bootstrap.env import make_gym_environment

    gymnasium = ModuleType("gymnasium")

    def make_vec(env_id: str, **kwargs: object) -> tuple[str, dict[str, object]]:
        return env_id, kwargs

    def make(env_id: str, **kwargs: object) -> object:
        return object()

    gymnasium.make = make  # type: ignore[attr-defined]
    gymnasium.make_vec = make_vec  # type: ignore[attr-defined]

    env = make_gym_environment(
        gymnasium,
        env_id="VectorEnv-v0",
        kwargs={"foo": "bar"},
        num_envs=3,
        vectorization_mode="async",
    )

    assert env == (
        "VectorEnv-v0",
        {"num_envs": 3, "foo": "bar", "vectorization_mode": "async"},
    )


def test_make_gym_environment_uses_vector_class_fallback() -> None:
    from rlmesh._bootstrap.env import make_gym_environment

    class AsyncVectorEnv:
        def __init__(self, factories: list[Callable[[], object]]) -> None:
            self.envs = [factory() for factory in factories]

    gymnasium = ModuleType("gymnasium")

    def make(env_id: str, **kwargs: object) -> tuple[str, dict[str, object]]:
        return env_id, kwargs

    gymnasium.make = make  # type: ignore[attr-defined]
    gymnasium.vector = SimpleNamespace(  # type: ignore[attr-defined]
        AsyncVectorEnv=AsyncVectorEnv
    )

    env = make_gym_environment(
        gymnasium,
        env_id="VectorEnv-v0",
        kwargs={"seeded": True},
        num_envs=2,
        vectorization_mode="async",
    )

    assert isinstance(env, AsyncVectorEnv)
    assert env.envs == [
        ("VectorEnv-v0", {"seeded": True}),
        ("VectorEnv-v0", {"seeded": True}),
    ]


def test_load_env_from_spec_dispatches_gym(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlmesh._bootstrap.env import load_env_from_spec

    gymnasium = ModuleType("gymnasium")

    def make(env_id: str, **kwargs: object) -> tuple[str, dict[str, object]]:
        return env_id, kwargs

    gymnasium.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    env = load_env_from_spec(
        {
            "kind": "gym",
            "env_id": "CartPole-v1",
            "kwargs": {"render_mode": "rgb_array"},
        }
    )

    assert env == ("CartPole-v1", {"render_mode": "rgb_array"})


def test_load_env_entrypoint_imports_packages_and_forwards_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.env import load_env_entrypoint

    imported: list[str] = []
    registration_module = ModuleType("fake_env_registration")
    env_module = ModuleType("fake_env_module")
    captured: dict[str, object] = {}

    def make_env(size: int) -> object:
        captured["size"] = size
        return SimpleNamespace(reset=lambda: None, step=lambda action: None)

    env_module.factories = SimpleNamespace(make=make_env)  # type: ignore[attr-defined]

    real_import = importlib.import_module

    def import_module(name: str) -> ModuleType:
        imported.append(name)
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setitem(sys.modules, "fake_env_registration", registration_module)
    monkeypatch.setitem(sys.modules, "fake_env_module", env_module)

    env = load_env_entrypoint(
        "fake_env_module:factories.make",
        ["fake_env_registration"],
        kwargs={"size": 3},
    )

    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert captured == {"size": 3}
    assert imported == ["fake_env_registration", "fake_env_module"]


def test_load_env_entrypoint_rejects_malformed_entrypoint() -> None:
    from rlmesh._bootstrap.env import load_env_entrypoint

    with pytest.raises(ValueError, match="module:callable"):
        load_env_entrypoint("fake_env_module")


def test_load_env_entrypoint_rejects_missing_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.env import load_env_entrypoint

    module = ModuleType("fake_env_module")
    monkeypatch.setitem(sys.modules, "fake_env_module", module)

    with pytest.raises(AttributeError, match="could not resolve"):
        load_env_entrypoint("fake_env_module:missing")


def test_load_env_entrypoint_rejects_non_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.env import load_env_entrypoint

    module = ModuleType("fake_env_module")
    module.value = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_env_module", module)

    with pytest.raises(TypeError, match="did not resolve to a callable"):
        load_env_entrypoint("fake_env_module:value")


def test_load_env_entrypoint_rejects_non_env_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.env import load_env_entrypoint

    module = ModuleType("fake_env_module")
    module.make_env = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_env_module", module)

    with pytest.raises(TypeError, match="did not return an environment"):
        load_env_entrypoint("fake_env_module:make_env")


def test_legacy_sandbox_bootstrap_shim_dispatches_gym(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import _sandbox_bootstrap

    gymnasium = ModuleType("gymnasium")

    def make(env_id: str, **kwargs: object) -> tuple[str, dict[str, object]]:
        return env_id, kwargs

    gymnasium.make = make  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    env = _sandbox_bootstrap.load_env({"kind": "gym", "env_id": "CartPole-v1"})

    assert env == ("CartPole-v1", {})


def test_normalize_hf_env_selects_suite() -> None:
    from rlmesh._bootstrap.env import normalize_hf_env

    selected = SimpleNamespace(reset=lambda: None, step=lambda action: None)

    assert (
        normalize_hf_env({"suite-a": object(), "suite-b": selected}, suite="suite-b")
        is selected
    )


def test_load_predict_resolves_nested_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlmesh._bootstrap.model import load_predict

    module = ModuleType("fake_model_module")
    module.policy = SimpleNamespace(  # type: ignore[attr-defined]
        nested=lambda observation: {"obs": observation}
    )
    monkeypatch.setitem(sys.modules, "fake_model_module", module)

    predict = load_predict("fake_model_module:policy.nested")

    assert predict(3) == {"obs": 3}


def test_parse_entrypoint_rejects_missing_callable() -> None:
    from rlmesh._bootstrap.model import parse_entrypoint

    with pytest.raises(ValueError, match="module:callable"):
        parse_entrypoint("fake_model_module")
