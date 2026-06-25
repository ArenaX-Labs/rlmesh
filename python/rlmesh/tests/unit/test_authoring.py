"""Authoring layer: ModelRecipe/EnvRecipe coercion, gates, and constructors.

The full ``Model(...).run(env)`` loop is unchanged and exercised elsewhere; these
tests pin the Phase-3 seam -- how a recipe is coerced into a model and how the
constructors run the lifecycle hooks.
"""

from __future__ import annotations

import pytest
import rlmesh
from rlmesh._authoring import EnvRecipe, ModelRecipe
from rlmesh._bootstrap.loaders import (
    construct_authored_env,
    construct_authored_model,
    looks_like_policy,
)
from rlmesh._models._eval import coerce_model


class _Policy(ModelRecipe):
    # a stand-in for a ModelSpec; coercion only reads the attribute by name.
    spec = "SPEC"  # pyright: ignore[reportAssignmentType]

    def __init__(self) -> None:
        self.loaded = False
        self.resets = 0

    def load(self) -> None:
        self.loaded = True

    def predict(self, observation: object) -> int:
        return 7

    def reset(self) -> None:
        self.resets += 1


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


def test_coerce_model_wires_recipe_into_the_policy_slot() -> None:
    coerced = coerce_model(_Policy, spec=None)
    assert isinstance(coerced.policy, _Policy)
    assert coerced.policy.loaded is True
    assert coerced.predict == coerced.policy.predict  # bound method of the instance
    assert coerced.spec == "SPEC"  # falls back to the recipe's spec
    # reset/close are the recipe's bound methods, so the run loop fires them.
    assert coerced.on_reset is not None
    assert coerced.on_reset == coerced.policy.reset
    coerced.on_reset()
    assert coerced.policy.resets == 1


def test_coerce_model_explicit_spec_overrides_recipe_spec() -> None:
    coerced = coerce_model(_Policy, spec="OVERRIDE")
    assert coerced.spec == "OVERRIDE"


def test_coerce_model_bare_callable_is_unchanged() -> None:
    fn = lambda obs: 0  # noqa: E731
    coerced = coerce_model(fn, spec=None)
    assert coerced.predict is fn
    assert coerced.policy is None
    assert coerced.on_reset is None


def test_coerce_model_rejects_non_callable_non_recipe() -> None:
    with pytest.raises(TypeError, match="predict callable or a ModelRecipe"):
        coerce_model(object(), spec=None)


def test_model_constructs_from_a_recipe_class() -> None:
    model = rlmesh.Model(_Policy)
    assert model.spec == "SPEC"  # recipe spec flows through Model


class _Env(EnvRecipe):
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


def test_authoring_bases_are_exported() -> None:
    assert rlmesh.ModelRecipe is ModelRecipe
    assert rlmesh.EnvRecipe is EnvRecipe
