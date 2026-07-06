"""Authoring layer: policy/env coercion, gates, and constructors.

The full ``Model(...).run(env)`` loop is unchanged and exercised elsewhere; these
tests pin the seam -- how a duck-typed policy or a ``Model`` subclass is coerced into
a model, how the serve dispatch avoids double-construction, and how the constructors
run the lifecycle hooks.
"""

from __future__ import annotations

import pytest
import rlmesh
from rlmesh._authoring import EnvFactory
from rlmesh._bootstrap.loaders import (
    construct_authored_env,
    construct_authored_model,
    looks_like_policy,
)
from rlmesh._models._coerce import coerce_model


class _Policy:
    """A duck-typed policy object (NOT a ``Model`` subclass): wrapped via coerce_model.

    rlmesh treats ``__init__`` as authoritative for a duck-typed policy and never
    calls ``load()`` (doing so would load weights twice). ``reset()`` is wired to the
    episode-END edge.
    """

    # a stand-in for a ModelSpec; coercion only reads the attribute by name.
    spec = "SPEC"

    def __init__(self) -> None:
        self.loads = 1  # constructs fully here
        self.episode_ends = 0

    def load(self) -> None:
        self.loads += 1  # must NOT be auto-called by rlmesh (would double-load)

    def predict(self, observation: object) -> int:
        return 7

    def reset(self) -> None:
        self.episode_ends += 1

    def close(self) -> None:
        pass


def test_looks_like_policy_gate() -> None:
    assert looks_like_policy(_Policy) is True  # class: predict is an unbound function
    assert looks_like_policy(_Policy()) is True  # instance: predict is a bound method
    assert looks_like_policy(lambda obs: 0) is False  # bare callable has no .predict


def test_construct_authored_model_instantiates_without_double_loading() -> None:
    # __init__ is authoritative for a duck-typed policy; rlmesh must NOT call load().
    inst = construct_authored_model(_Policy)
    assert isinstance(inst, _Policy)
    assert inst.loads == 1
    assert inst.predict(None) == 7


def test_construct_authored_model_accepts_an_instance() -> None:
    given = _Policy()
    inst = construct_authored_model(given)
    assert inst is given
    assert inst.loads == 1  # not re-loaded


def test_coerce_model_wires_policy_into_the_policy_slot() -> None:
    coerced = coerce_model(_Policy, spec=None)
    assert isinstance(coerced.policy, _Policy)
    assert coerced.policy.loads == 1  # __init__ only; no auto load()
    assert coerced.predict == coerced.policy.predict  # bound method of the instance
    assert coerced.spec == "SPEC"  # falls back to the policy's spec
    # reset/close are the policy's bound methods; reset() fires at the episode-END
    # edge (on_episode_end), the same on local and served paths.
    assert coerced.on_episode_end is not None
    assert coerced.on_episode_end == coerced.policy.reset
    coerced.on_episode_end()
    assert coerced.policy.episode_ends == 1


def test_coerce_model_explicit_spec_overrides_policy_spec() -> None:
    coerced = coerce_model(_Policy, spec="OVERRIDE")
    assert coerced.spec == "OVERRIDE"


def test_coerce_model_bare_callable_is_unchanged() -> None:
    fn = lambda obs: 0  # noqa: E731
    coerced = coerce_model(fn, spec=None)
    assert coerced.predict is fn
    assert coerced.policy is None
    assert coerced.on_episode_end is None
    assert coerced.predict_chunk is None
    assert coerced.predict_batch is None
    assert coerced.predict_chunk_batch is None


def test_duck_policy_predict_chunk_is_picked_up_and_actually_chunks() -> None:
    # A duck-typed policy's chunk corner must survive coercion: with
    # execution_horizon=3 the replay calls predict_chunk once per 3 steps
    # instead of silently dropping the corner and re-planning every step.
    calls = {"chunk": 0, "predict": 0}

    class _ChunkPolicy:
        def predict(self, observation: object) -> int:
            calls["predict"] += 1
            return 0

        def predict_chunk(self, observation: object) -> list[int]:
            calls["chunk"] += 1
            return [0, 1, 2]

    class _SixStepEnv:
        def __init__(self) -> None:
            from rlmesh import spaces

            self._steps = 0
            self.observation_space = spaces.Discrete(1)
            self.action_space = spaces.Discrete(3)

        def reset(
            self, *, seed: object = None, options: object = None
        ) -> tuple[int, dict[str, object]]:
            self._steps = 0
            return 0, {}

        def step(
            self, action: object
        ) -> tuple[int, float, bool, bool, dict[str, object]]:
            self._steps += 1
            return 0, 0.0, self._steps >= 6, False, {}

        def close(self) -> None:
            pass

    coerced = coerce_model(_ChunkPolicy, spec=None)
    assert coerced.predict_chunk is not None

    result = rlmesh.run(_ChunkPolicy(), _SixStepEnv(), execution_horizon=3)
    assert result.total_steps == 6
    assert calls["chunk"] == 2  # re-planned every 3 steps
    assert calls["predict"] == 0  # the chunk corner drove the whole episode


def test_coerce_model_rejects_non_callable_non_policy() -> None:
    with pytest.raises(TypeError, match="predict callable or a policy object"):
        coerce_model(object(), spec=None)


def test_model_constructs_from_a_duck_typed_policy_class() -> None:
    model = rlmesh.Model(_Policy)
    assert model.spec == "SPEC"  # policy spec flows through Model


# --- Model subclass authoring (the merged ModelRecipe path) ---


class _ModelPolicy(rlmesh.Model):
    spec = "SPEC"  # pyright: ignore[reportAssignmentType]

    def load(self) -> None:
        self.loaded = True

    def predict(self, observation: object) -> int:
        return 7


def test_model_subclass_loads_once_and_exposes_spec() -> None:
    model = _ModelPolicy()
    assert model.loaded is True  # load() fired during __init__
    assert model.spec == "SPEC"  # class-attribute spec resolved onto the instance


def test_model_subclass_spec_kwarg_overrides_class_attr() -> None:
    model = _ModelPolicy(spec="OVERRIDE")
    assert model.spec == "OVERRIDE"


def test_model_rejects_a_model_as_source() -> None:
    # A Model builds its own worker; wrapping it again would double-construct. The
    # guard lives at the construction gateway (ModelBase.__init__), not in coerce.
    with pytest.raises(TypeError, match="Instantiate your Model subclass"):
        rlmesh.Model(_ModelPolicy)
    with pytest.raises(TypeError, match="Instantiate your Model subclass"):
        rlmesh.Model(_ModelPolicy())


def test_model_subclass_serve_loads_then_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the blocking terminal is stubbed; subclass __init__ (load + worker) runs
    # for real, so this fails if serve ever regresses to a no-op.
    from rlmesh._models.base import ModelBase

    seen: dict[str, object] = {}

    def fake_serve(self: object, address: str, *, options=None) -> None:
        seen["address"] = address

    monkeypatch.setattr(ModelBase, "serve", fake_serve)
    model = _ModelPolicy()
    model.serve("127.0.0.1:5555")
    assert model.loaded is True  # load() fired during construction
    assert seen == {"address": "127.0.0.1:5555"}


# --- serve dispatch: resolve a model source without double-construction ---


def test_serve_model_dispatch_avoids_double_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # serve_model resolves a source to one served Model: a subclass *class* is
    # instantiated once (load fires), an instance is served as-is -- never re-wrapped.
    from rlmesh import serve
    from rlmesh._models.base import ModelBase

    served: list[object] = []

    def fake_serve(self: object, address: str, *, options=None) -> None:
        served.append(self)

    monkeypatch.setattr(ModelBase, "serve", fake_serve)

    serve.serve_model(_ModelPolicy, "127.0.0.1:5555")
    assert len(served) == 1
    assert isinstance(served[0], _ModelPolicy)
    assert served[0].loaded is True  # load() ran exactly once at instantiation

    served.clear()
    inst = _ModelPolicy()
    serve.serve_model(inst, "127.0.0.1:5555")
    assert served == [inst]  # existing instance served as-is, not re-wrapped


# --- env authoring (unchanged) ---


class _Env(EnvFactory):
    tags = None

    def __init__(self) -> None:
        self.prepared = False

    def prepare(self) -> None:
        self.prepared = True

    def make(self, **kwargs: object) -> object:
        return ("env", self.prepared, kwargs)


def test_construct_authored_env_prepares_then_makes() -> None:
    env = construct_authored_env(_Env, render_mode="rgb_array")
    assert env == ("env", True, {"render_mode": "rgb_array"})


def test_env_recipe_serve_prepares_makes_and_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class FakeEnvServer:
        def __init__(
            self,
            env: object,
            address: str,
            *,
            tags: object = None,
            framework: object = None,
            device: object = None,
        ) -> None:
            _ = framework, device  # accepted (neutral server), not asserted here
            self.address = address
            seen.update(env=env, address=address, tags=tags)

        def serve(self) -> None:
            seen["served"] = True

    monkeypatch.setattr(rlmesh, "EnvServer", FakeEnvServer)
    env = _Env()
    env.serve("127.0.0.1:5555", render_mode="rgb_array")
    assert env.prepared is True  # prepare() fired; serve is no longer a no-op
    assert seen == {
        "env": ("env", True, {"render_mode": "rgb_array"}),
        "address": "127.0.0.1:5555",
        "tags": None,
        "served": True,
    }


def test_env_factory_serve_separates_serving_options_from_make_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The serving options are named in EnvFactory.serve's signature, so they can
    # never silently mix with make kwargs on the way into serve_env.
    from rlmesh import serve as serve_mod

    seen: dict[str, object] = {}

    def fake_serve_env(env_source: object, address: str, /, **kwargs: object) -> None:
        seen["address"] = address
        seen.update(kwargs)

    monkeypatch.setattr(serve_mod, "serve_env", fake_serve_env)
    _Env().serve("127.0.0.1:5555", num_envs=2, framework="numpy", task_id=7)
    assert seen == {
        "address": "127.0.0.1:5555",
        "num_envs": 2,
        "vectorization_mode": None,
        "framework": "numpy",
        "device": None,
        "task_id": 7,
    }


def test_authoring_bases_are_exported() -> None:
    assert rlmesh.EnvFactory is EnvFactory


# --- regression: model binding must not be silently swallowed ----------------


def test_construct_authored_model_rejects_swallowed_binding() -> None:
    # A Model that does not override load() has nowhere to apply a binding; the
    # default no-op load would swallow it silently. Fail loud instead.
    class _NoLoad(rlmesh.Model):
        def predict(self, observation: object) -> int:
            return 0

    with pytest.raises(TypeError, match="does not override load"):
        construct_authored_model(_NoLoad, checkpoint="x")


def test_construct_authored_model_applies_binding_via_load() -> None:
    seen: dict[str, object] = {}

    class _Loads(rlmesh.Model):
        def load(self, *, checkpoint: str = "default") -> None:
            seen["checkpoint"] = checkpoint

        def predict(self, observation: object) -> int:
            return 0

    construct_authored_model(_Loads, checkpoint="x")
    assert seen["checkpoint"] == "x"
