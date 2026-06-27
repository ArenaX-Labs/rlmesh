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
from rlmesh._models._eval import coerce_model


class _Policy:
    """A duck-typed policy object (NOT a ``Model`` subclass): wrapped via coerce_model."""

    # a stand-in for a ModelSpec; coercion only reads the attribute by name.
    spec = "SPEC"

    def __init__(self) -> None:
        self.loaded = False
        self.resets = 0

    def load(self) -> None:
        self.loaded = True

    def predict(self, observation: object) -> int:
        return 7

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass


def test_looks_like_policy_gate() -> None:
    assert looks_like_policy(_Policy) is True  # class: predict is an unbound function
    assert looks_like_policy(_Policy()) is True  # instance: predict is a bound method
    assert looks_like_policy(lambda obs: 0) is False  # bare callable has no .predict


def test_construct_authored_model_instantiates_and_loads() -> None:
    inst = construct_authored_model(_Policy)
    assert isinstance(inst, _Policy)
    assert inst.loaded is True
    assert inst.predict(None) == 7


def test_construct_authored_model_accepts_an_instance() -> None:
    given = _Policy()
    inst = construct_authored_model(given)
    assert inst is given
    assert inst.loaded is True


def test_coerce_model_wires_policy_into_the_policy_slot() -> None:
    coerced = coerce_model(_Policy, spec=None)
    assert isinstance(coerced.policy, _Policy)
    assert coerced.policy.loaded is True
    assert coerced.predict == coerced.policy.predict  # bound method of the instance
    assert coerced.spec == "SPEC"  # falls back to the policy's spec
    # reset/close are the policy's bound methods, so the run loop fires them.
    assert coerced.on_reset is not None
    assert coerced.on_reset == coerced.policy.reset
    coerced.on_reset()
    assert coerced.policy.resets == 1


def test_coerce_model_explicit_spec_overrides_policy_spec() -> None:
    coerced = coerce_model(_Policy, spec="OVERRIDE")
    assert coerced.spec == "OVERRIDE"


def test_coerce_model_bare_callable_is_unchanged() -> None:
    fn = lambda obs: 0  # noqa: E731
    coerced = coerce_model(fn, spec=None)
    assert coerced.predict is fn
    assert coerced.policy is None
    assert coerced.on_reset is None


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


def test_coerce_model_rejects_a_model_subclass() -> None:
    # A Model builds its own worker; wrapping it again would double-construct.
    with pytest.raises(TypeError, match="Instantiate your Model subclass"):
        coerce_model(_ModelPolicy, spec=None)
    with pytest.raises(TypeError, match="Instantiate your Model subclass"):
        coerce_model(_ModelPolicy(), spec=None)


def test_model_subclass_serve_loads_then_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only the blocking terminal is stubbed; subclass __init__ (load + worker) runs
    # for real, so this fails if serve ever regresses to a no-op.
    from rlmesh._models.base import ModelBase

    seen: dict[str, object] = {}

    def fake_serve(
        self: object, address: str, *, token: str = "", options=None
    ) -> None:
        seen["address"] = address
        seen["token"] = token

    monkeypatch.setattr(ModelBase, "serve", fake_serve)
    model = _ModelPolicy()
    model.serve("127.0.0.1:5555", token="tk")
    assert model.loaded is True  # load() fired during construction
    assert seen == {"address": "127.0.0.1:5555", "token": "tk"}


# --- serve dispatch: resolve a model source without double-construction ---


def test_serve_model_dispatch_avoids_double_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # serve_model resolves a source to one served Model: a subclass *class* is
    # instantiated once (load fires), an instance is served as-is -- never re-wrapped.
    from rlmesh import serve
    from rlmesh._models.base import ModelBase

    served: list[object] = []

    def fake_serve(
        self: object, address: str, *, token: str = "", options=None
    ) -> None:
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
        def __init__(self, env: object, address: str, *, tags: object = None) -> None:
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
