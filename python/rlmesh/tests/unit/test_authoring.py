"""Authoring layer: policy/env coercion, gates, and constructors.

The full ``Model(...).run(env)`` loop is unchanged and exercised elsewhere; these
tests pin the seam -- how a duck-typed policy or a ``Model`` subclass is coerced into
a model, how the serve dispatch avoids double-construction, and how the constructors
run the lifecycle hooks.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import rlmesh
import rlmesh.numpy
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

    result = rlmesh.run(
        rlmesh.numpy.Model(_ChunkPolicy()), _SixStepEnv(), execution_horizon=3
    )
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
            options: object = None,
        ) -> None:
            # Accepted (neutral server), not asserted here.
            _ = framework, device, options
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
        "workflow_edition": None,
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


# --- contract branches: tag_params / tags_for ---------------------------------


def _tags(role: str) -> Any:
    import rlmesh.adapters as adapt

    return adapt.EnvTags(
        observation={"eef": adapt.StateTag(adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(role, dim=3)),
    )


_DELTA = _tags("action/delta_eef_pos")
_ABSOLUTE = _tags("action/eef_pos")


class _BranchEnv:
    """A make() return with a *class-level* metadata dict (the gymnasium shape)."""

    metadata: ClassVar[dict[str, object]] = {}

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed


class _Branched(EnvFactory):
    tags = _DELTA
    params = rlmesh.ParamSpec(
        rlmesh.Param("action_type", type="enum", choices=("delta", "abs")),
        rlmesh.Param("seed", type="int"),
    )
    tag_params = ("action_type",)

    @classmethod
    def tags_for(cls, **params: object) -> object:
        return _DELTA if params["action_type"] == "delta" else _ABSOLUTE

    def make(self, action_type: str = "delta", seed: int = 0) -> _BranchEnv:
        return _BranchEnv(seed)


def _published(env: object) -> tuple[object, object]:
    from rlmesh.adapters import ENV_BRANCH_METADATA_KEY, EnvTags

    metadata = cast("dict[str, Any]", env.metadata)  # type: ignore[attr-defined]
    return metadata.get(ENV_BRANCH_METADATA_KEY), EnvTags.from_metadata(metadata)


def test_default_branch_stamps_the_signature_defaults() -> None:
    # Nothing supplied: the binding is still complete (make's defaults applied),
    # and the tags are the default branch's -- i.e. the factory's own ``tags``.
    branch, tags = _published(_Branched().make())
    assert branch == {"action_type": "delta"}
    assert tags == _DELTA
    # The stamp is an instance attribute: the class-level dict is untouched, so
    # one branched env never leaks its branch onto the next.
    assert _BranchEnv.metadata == {}


def test_branch_binds_a_positional_discriminant() -> None:
    # bind() reads make's own signature, so a positional value selects the branch
    # exactly as the keyword form does.
    assert _published(_Branched().make("abs"))[0] == {"action_type": "abs"}
    assert _published(_Branched().make(action_type="abs"))[1] == _ABSOLUTE


def test_branch_rejects_a_value_outside_the_declared_choices() -> None:
    from rlmesh.params import ParamError

    # Pre-construction: the bind runs before make()'s body.
    with pytest.raises(ParamError, match="action_type='ee'"):
        _Branched().make(action_type="ee")


def test_unbranched_factory_stamps_no_branch_key() -> None:
    from rlmesh.adapters import ENV_BRANCH_METADATA_KEY

    class _Plain(EnvFactory):
        tags = _DELTA

        def make(self, seed: int = 0) -> _BranchEnv:
            return _BranchEnv(seed)

    env = _Plain().make()
    assert ENV_BRANCH_METADATA_KEY not in cast("dict[str, Any]", env.metadata)


# The eight class-creation TypeErrors. Each builds a factory that is wrong in
# exactly one way; the class statement itself is what must fail.


def test_tag_params_rejects_an_undeclared_discriminant() -> None:
    with pytest.raises(TypeError, match="not declared in params"):

        class _Undeclared(EnvFactory):
            tags = _DELTA
            tag_params = ("action_type",)

            def make(self, action_type: str = "delta") -> _BranchEnv:
                return _BranchEnv()


def test_tag_params_rejects_a_non_enumerable_discriminant() -> None:
    with pytest.raises(TypeError, match="no choices"):

        class _Open(EnvFactory):
            tags = _DELTA
            params = rlmesh.ParamSpec(rlmesh.Param("action_type", type="str"))
            tag_params = ("action_type",)

            def make(self, action_type: str = "delta") -> _BranchEnv:
                return _BranchEnv()


def test_tag_params_rejects_a_discriminant_without_a_signature_default() -> None:
    spec = rlmesh.ParamSpec(
        rlmesh.Param("action_type", type="enum", choices=("delta", "abs"))
    )
    with pytest.raises(TypeError, match="signature default"):

        class _NoDefault(EnvFactory):
            tags = _DELTA
            params = spec
            tag_params = ("action_type",)

            def make(self, action_type: str) -> _BranchEnv:
                return _BranchEnv()

    # A default that is not one of the choices is the same defect: the default
    # branch would not be in the table the label publishes.
    with pytest.raises(TypeError, match="signature default"):

        class _OffChoices(EnvFactory):
            tags = _DELTA
            params = spec
            tag_params = ("action_type",)

            def make(self, action_type: str = "ee") -> _BranchEnv:
                return _BranchEnv()


def test_tag_params_rejects_a_product_over_the_branch_cap() -> None:
    with pytest.raises(TypeError, match="over the limit of 8"):

        class _TooMany(EnvFactory):
            tags = _DELTA
            params = rlmesh.ParamSpec(
                rlmesh.Param("a", type="enum", choices=("x", "y", "z")),
                rlmesh.Param("b", type="enum", choices=(1, 2, 3)),
            )
            tag_params = ("a", "b")

            def make(self, a: str = "x", b: int = 1) -> _BranchEnv:
                return _BranchEnv()


def test_tags_for_override_without_tag_params_is_rejected() -> None:
    with pytest.raises(TypeError, match="declares no tag_params"):

        class _Orphan(EnvFactory):
            tags = _DELTA

            @classmethod
            def tags_for(cls, **params: object) -> object:
                return _ABSOLUTE

            def make(self) -> _BranchEnv:
                return _BranchEnv()


def test_tags_for_returning_a_non_envtags_is_rejected() -> None:
    with pytest.raises(TypeError, match="not EnvTags or None"):

        class _WrongType(EnvFactory):
            tags = _DELTA
            params = rlmesh.ParamSpec(
                rlmesh.Param("action_type", type="enum", choices=("delta", "abs"))
            )
            tag_params = ("action_type",)

            @classmethod
            def tags_for(cls, **params: object) -> object:
                return _DELTA if params["action_type"] == "delta" else "absolute"

            def make(self, action_type: str = "delta") -> _BranchEnv:
                return _BranchEnv()


def test_mixed_adapted_and_generic_branches_are_rejected() -> None:
    with pytest.raises(TypeError, match="adapted or generic, not both"):

        class _Mixed(EnvFactory):
            tags = _DELTA
            params = rlmesh.ParamSpec(
                rlmesh.Param("action_type", type="enum", choices=("delta", "abs"))
            )
            tag_params = ("action_type",)

            @classmethod
            def tags_for(cls, **params: object) -> object:
                return _DELTA if params["action_type"] == "delta" else None

            def make(self, action_type: str = "delta") -> _BranchEnv:
                return _BranchEnv()


def test_tags_disagreeing_with_the_default_branch_is_rejected() -> None:
    with pytest.raises(TypeError, match="its default branch"):

        class _Disagrees(EnvFactory):
            tags = _DELTA  # ...but the default branch answers with _ABSOLUTE
            params = rlmesh.ParamSpec(
                rlmesh.Param("action_type", type="enum", choices=("delta", "abs"))
            )
            tag_params = ("action_type",)

            @classmethod
            def tags_for(cls, **params: object) -> object:
                return _ABSOLUTE if params["action_type"] == "delta" else _DELTA

            def make(self, action_type: str = "delta") -> _BranchEnv:
                return _BranchEnv()


def test_tag_params_without_a_tags_for_override_is_a_spaces_only_branch() -> None:
    # Legal: the discriminant moves the spaces, not the contract.
    class _SpacesOnly(EnvFactory):
        tags = _DELTA
        params = rlmesh.ParamSpec(
            rlmesh.Param("width", type="enum", choices=(128, 256))
        )
        tag_params = ("width",)

        def make(self, width: int = 256) -> _BranchEnv:
            return _BranchEnv()

    branch, tags = _published(_SpacesOnly().make(width=128))
    assert branch == {"width": 128}
    assert tags == _DELTA


def test_tag_branches_orders_the_default_first() -> None:
    from rlmesh._authoring import _tag_branches

    assert _tag_branches(_Branched) == [
        {"action_type": "delta"},
        {"action_type": "abs"},
    ]
