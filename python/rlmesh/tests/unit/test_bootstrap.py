from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest


def test_load_environment_imports_registration_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._bootstrap.loaders import load_environment

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
    from rlmesh._bootstrap.loaders import load_environment

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
    from rlmesh._bootstrap.gym_support import make_gym_environment

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
    from rlmesh._bootstrap.gym_support import EpisodeSeedEnv, make_gym_environment

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
    lanes = cast("list[EpisodeSeedEnv]", env.envs)
    assert all(isinstance(lane, EpisodeSeedEnv) for lane in lanes)
    assert [lane._env for lane in lanes] == [
        ("VectorEnv-v0", {"seeded": True}),
        ("VectorEnv-v0", {"seeded": True}),
    ]


def test_load_env_from_spec_dispatches_gym(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlmesh._bootstrap.loaders import load_env_from_spec

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
    from rlmesh._bootstrap.loaders import load_env_entrypoint

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


@pytest.mark.parametrize(
    ("attr", "entrypoint", "exc", "match"),
    [
        (None, "fake_env_module", "EntrypointConstructionError", "module:callable"),
        (
            None,
            "fake_env_module:missing",
            "EntrypointConstructionError",
            "could not resolve",
        ),
        (
            object(),
            "fake_env_module:value",
            "EntrypointConstructionError",
            "did not resolve to a callable",
        ),
        (
            (lambda: object()),
            "fake_env_module:make_env",
            "TypeError",
            "did not return an environment",
        ),
    ],
)
def test_load_env_entrypoint_rejects(
    monkeypatch: pytest.MonkeyPatch,
    attr: object,
    entrypoint: str,
    exc: str,
    match: str,
) -> None:
    from rlmesh._bootstrap.loaders import (
        EntrypointConstructionError,
        load_env_entrypoint,
    )

    module = ModuleType("fake_env_module")
    if attr is not None:
        setattr(module, entrypoint.split(":", 1)[1], attr)
    monkeypatch.setitem(sys.modules, "fake_env_module", module)

    expected = TypeError if exc == "TypeError" else EntrypointConstructionError
    with pytest.raises(expected, match=match):
        load_env_entrypoint(entrypoint)


def test_normalize_hf_env_returns_direct_env() -> None:
    from rlmesh._bootstrap.loaders import normalize_hf_env

    selected = SimpleNamespace(reset=lambda: None, step=lambda action: None)

    assert normalize_hf_env(selected, suite=None, task=None) is selected


def test_normalize_hf_env_selects_suite() -> None:
    from rlmesh._bootstrap.loaders import normalize_hf_env

    selected = SimpleNamespace(reset=lambda: None, step=lambda action: None)

    assert (
        normalize_hf_env(
            {"suite-a": object(), "suite-b": selected},
            suite="suite-b",
            task=None,
        )
        is selected
    )


def test_normalize_hf_env_auto_selects_only_nested_task() -> None:
    from rlmesh._bootstrap.loaders import normalize_hf_env

    selected = SimpleNamespace(reset=lambda: None, step=lambda action: None)

    assert (
        normalize_hf_env({"cartpole_suite": {0: selected}}, suite=None, task=None)
        is selected
    )


def test_normalize_hf_env_selects_nested_task_by_string_key() -> None:
    from rlmesh._bootstrap.loaders import normalize_hf_env

    selected = SimpleNamespace(reset=lambda: None, step=lambda action: None)

    assert (
        normalize_hf_env(
            {"cartpole_suite": {0: selected, 1: object()}},
            suite="cartpole_suite",
            task="0",
        )
        is selected
    )


@pytest.mark.parametrize(
    ("bundle", "suite", "match"),
    [
        ({"suite-a": object(), "suite-b": object()}, None, "suite-a, suite-b"),
        ({"cartpole_suite": {0: object(), 1: object()}}, "cartpole_suite", "0, 1"),
    ],
)
def test_normalize_hf_env_lists_ambiguous_choices(
    bundle: object, suite: str | None, match: str
) -> None:
    from rlmesh._bootstrap.loaders import normalize_hf_env

    with pytest.raises(ValueError, match=match):
        normalize_hf_env(bundle, suite=suite, task=None)


def test_load_hf_env_passes_task_from_bootstrap_spec(tmp_path) -> None:
    from rlmesh._bootstrap.loaders import load_hf_env

    source = tmp_path / "source"
    source.mkdir()
    (source / "env.py").write_text(
        """
class TinyEnv:
    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0.0, True, False, {}


def make_env(**kwargs):
    return {"cartpole_suite": {0: object(), 1: TinyEnv()}}
""",
        encoding="utf-8",
    )

    env = load_hf_env(
        {
            "kind": "hf",
            "source_subdir": str(source),
            "suite": "cartpole_suite",
            "task": "1",
        }
    )

    assert hasattr(env, "reset")
    assert hasattr(env, "step")


def test_load_predict_resolves_nested_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    from rlmesh._bootstrap.loaders import load_predict

    module = ModuleType("fake_model_module")
    module.policy = SimpleNamespace(  # type: ignore[attr-defined]
        nested=lambda observation: {"obs": observation}
    )
    monkeypatch.setitem(sys.modules, "fake_model_module", module)

    predict = load_predict("fake_model_module:policy.nested")

    assert predict(3) == {"obs": 3}


def test_parse_entrypoint_rejects_missing_callable() -> None:
    from rlmesh._entrypoint import parse_entrypoint

    with pytest.raises(ValueError, match="module:callable"):
        parse_entrypoint("fake_model_module")


def test_recipe_construction_error_is_catchable_as_import_error() -> None:
    # load_env_entrypoint is public (rlmesh._serving.load_env_entrypoint) and used to
    # raise a raw ImportError/AttributeError/TypeError/ValueError. The wrapper must
    # stay catchable by an old-style `except ImportError` so callers do not break.
    from rlmesh._bootstrap.loaders import EntrypointConstructionError

    assert issubclass(EntrypointConstructionError, ImportError)
    assert issubclass(EntrypointConstructionError, RuntimeError)


def test_load_env_entrypoint_malformed_caught_as_import_error() -> None:
    # A bad entrypoint that previously surfaced a raw ImportError must still be
    # catchable as ImportError even though it is now a EntrypointConstructionError.
    from rlmesh._bootstrap.loaders import (
        EntrypointConstructionError,
        load_env_entrypoint,
    )

    with pytest.raises(ImportError) as excinfo:
        load_env_entrypoint("fake_env_module")
    assert isinstance(excinfo.value, EntrypointConstructionError)


def test_load_env_entrypoint_does_not_wrap_factory_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # EntrypointConstructionError wraps ONLY the import/resolve boundary; an error
    # raised inside a successfully-resolved factory must propagate raw.
    from rlmesh._bootstrap.loaders import (
        EntrypointConstructionError,
        load_env_entrypoint,
    )

    module = ModuleType("fake_env_module")

    def boom(**_kwargs: object) -> object:
        raise RuntimeError("boom-from-make")

    module.boom = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_env_module", module)

    with pytest.raises(RuntimeError, match="boom-from-make") as excinfo:
        load_env_entrypoint("fake_env_module:boom")
    assert not isinstance(excinfo.value, EntrypointConstructionError)


def test_resolve_bootstrap_spec_reads_inline_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run path delivers a gym/hf spec inline via RLMESH_BOOTSTRAP_JSON.
    from rlmesh._bootstrap.spec_resolution import resolve_bootstrap_spec

    monkeypatch.setenv(
        "RLMESH_BOOTSTRAP_JSON", '{"spec":{"kind":"gym","env_id":"E-v0"}}'
    )

    spec = resolve_bootstrap_spec([], prog="x")

    assert spec["kind"] == "gym"


def test_load_environment_chains_lookup_error_with_suggestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gymnasium's NameNotFound ("Did you mean ...?") must be quoted and
    chained, not masked by a complaint about the missing gym module."""
    from rlmesh._bootstrap.loaders import load_environment

    name_not_found = type("NameNotFound", (Exception,), {})

    gymnasium = ModuleType("gymnasium")

    def missing_make(env_id: str, **kwargs: object) -> object:
        raise name_not_found(
            f"Environment {env_id} doesn't exist. Did you mean CartPole-v1?"
        )

    gymnasium.make = missing_make  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gym", None)  # type: ignore[arg-type]

    with pytest.raises(ImportError) as excinfo:
        load_environment("CartPole-v2", [], num_envs=1)

    message = str(excinfo.value)
    assert "Did you mean CartPole-v1?" in message
    assert "Missing modules" not in message
    assert isinstance(excinfo.value.__cause__, name_not_found)


def test_import_packages_chains_and_quotes_cause() -> None:
    from rlmesh._bootstrap.loaders import import_packages

    with pytest.raises(ImportError, match="Unable to import package") as excinfo:
        import_packages(["definitely_not_a_real_pkg_xyz"])

    assert "No module named" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_load_gym_env_does_not_retry_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A construction error (not a lookup miss) raises as-is instead of
    re-running side-effecting construction against gym and blaming both."""
    from rlmesh._bootstrap.loaders import load_gym_env

    gymnasium = ModuleType("gymnasium")

    def exploding_make(env_id: str, **kwargs: object) -> object:
        raise RuntimeError("construction exploded")

    gymnasium.make = exploding_make  # type: ignore[attr-defined]

    gym = ModuleType("gym")
    gym.make = lambda env_id, **k: pytest.fail("gym must not be retried")  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gym", gym)

    with pytest.raises(RuntimeError, match="construction exploded"):
        load_gym_env({"kind": "gym", "env_id": "E-v0"})


def test_vectorize_rejects_unknown_mode() -> None:
    from rlmesh._bootstrap.gym_support import vectorize

    with pytest.raises(ValueError, match="vectorization_mode"):
        vectorize(lambda: object(), 2, "Async")


def test_episode_seed_env_seeds_autoreset_rolls_deterministically() -> None:
    from rlmesh._bootstrap.gym_support import EpisodeSeedEnv

    class Env:
        def reset(self, *, seed: int | None = None, options: object = None):
            return seed, {}

    env = EpisodeSeedEnv(Env())
    assert env.reset(seed=7) == (7, {"seed": 7})
    rolls = [env.reset() for _ in range(2)]
    assert rolls == [(8, {"seed": 8}), (9, {"seed": 9})]
    assert env.reset(seed=7) == (7, {"seed": 7})
    assert env.reset() == (8, {"seed": 8})


def test_expect_vectorization_mode_none_means_auto() -> None:
    from rlmesh._bootstrap.spec_resolution import expect_vectorization_mode

    assert expect_vectorization_mode(None, "x") is None
    assert expect_vectorization_mode("async", "x") == "async"
    with pytest.raises(ValueError, match="'sync' or 'async'"):
        expect_vectorization_mode("Async", "x")


def test_call_hf_make_env_leaves_kwargs_only_signature_alone_by_default() -> None:
    """A **kwargs-only make_env only receives n_envs/use_async_envs when
    vectorization is actually requested, so the default single-env case never
    crashes a natural passthrough signature."""
    from rlmesh._bootstrap.gym_support import call_hf_make_env

    captured: dict[str, object] = {}

    def make_env(**kwargs: object) -> object:
        captured.clear()
        captured.update(kwargs)
        return SimpleNamespace(reset=lambda: None, step=lambda action: None)

    call_hf_make_env(make_env, {}, num_envs=1, vectorization_mode=None)
    assert captured == {}

    call_hf_make_env(make_env, {}, num_envs=3, vectorization_mode="async")
    assert captured == {"n_envs": 3, "use_async_envs": True}


def test_load_hf_env_uses_unique_module_names(tmp_path) -> None:
    """Two hf env loads in one process must not clobber each other's
    sys.modules entry."""
    from rlmesh._bootstrap.loaders import load_hf_env

    markers: dict[str, str] = {}
    for marker in ("alpha", "beta"):
        source = tmp_path / marker
        source.mkdir()
        (source / "env.py").write_text(
            f"""
class TinyEnv:
    marker = {marker!r}

    def reset(self, seed=None, options=None):
        return 0, {{}}

    def step(self, action):
        return 0, 0.0, True, False, {{}}


def make_env(**kwargs):
    return TinyEnv()
""",
            encoding="utf-8",
        )
        env = load_hf_env({"kind": "hf", "source_subdir": str(source)})
        markers[marker] = cast("Any", env).marker

    assert markers == {"alpha": "alpha", "beta": "beta"}
    hf_modules = [name for name in sys.modules if name.startswith("rlmesh_hf_env")]
    assert len(hf_modules) >= 2
