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
    spec = "SPEC"

    def __init__(self) -> None:
        self.loads = 1
        self.episode_ends = 0

    def load(self) -> None:
        self.loads += 1

    def predict(self, observation: object) -> int:
        return 7

    def reset(self) -> None:
        self.episode_ends += 1

    def close(self) -> None:
        pass


def test_looks_like_policy_gate() -> None:
    assert looks_like_policy(_Policy) is True
    assert looks_like_policy(_Policy()) is True
    assert looks_like_policy(lambda obs: 0) is False


def test_construct_authored_model_instantiates_without_double_loading() -> None:
    inst = construct_authored_model(_Policy)
    assert isinstance(inst, _Policy)
    assert inst.loads == 1
    assert inst.predict(None) == 7


def test_construct_authored_model_accepts_an_instance() -> None:
    given = _Policy()
    inst = construct_authored_model(given)
    assert inst is given
    assert inst.loads == 1


def test_coerce_model_wires_policy_into_the_policy_slot() -> None:
    coerced = coerce_model(_Policy, spec=None)
    assert isinstance(coerced.policy, _Policy)
    assert coerced.policy.loads == 1
    assert coerced.predict == coerced.policy.predict
    assert coerced.spec == "SPEC"
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
    assert calls["chunk"] == 2
    assert calls["predict"] == 0


def test_coerce_model_rejects_non_callable_non_policy() -> None:
    with pytest.raises(TypeError, match="predict callable or a policy object"):
        coerce_model(object(), spec=None)


def test_model_constructs_from_a_duck_typed_policy_class() -> None:
    model = rlmesh.Model(_Policy)
    assert model.spec == "SPEC"


class _ModelPolicy(rlmesh.Model):
    spec = "SPEC"  # pyright: ignore[reportAssignmentType]

    def load(self) -> None:
        self.loaded = True

    def predict(self, observation: object) -> int:
        return 7


def test_model_subclass_loads_once_and_exposes_spec() -> None:
    model = _ModelPolicy()
    assert model.loaded is True
    assert model.spec == "SPEC"


def test_model_subclass_spec_kwarg_overrides_class_attr() -> None:
    model = _ModelPolicy(spec="OVERRIDE")
    assert model.spec == "OVERRIDE"


def test_model_rejects_a_model_as_source() -> None:
    with pytest.raises(TypeError, match="Instantiate your Model subclass"):
        rlmesh.Model(_ModelPolicy)
    with pytest.raises(TypeError, match="Instantiate your Model subclass"):
        rlmesh.Model(_ModelPolicy())


def test_model_subclass_serve_loads_then_serves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh._models.base import ModelBase

    seen: dict[str, object] = {}

    def fake_serve(
        self: object, address: str, *, name: str, options=None, batching=None
    ) -> None:
        seen["address"] = address

    monkeypatch.setattr(ModelBase, "serve", fake_serve)
    model = _ModelPolicy()
    model.serve("127.0.0.1:5555", name="policy")
    assert model.loaded is True
    assert seen == {"address": "127.0.0.1:5555"}


def test_serve_model_dispatch_avoids_double_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rlmesh import serve
    from rlmesh._models.base import ModelBase

    served: list[object] = []

    def fake_serve(
        self: object, address: str, *, name: str, options=None, batching=None
    ) -> None:
        served.append(self)

    monkeypatch.setattr(ModelBase, "serve", fake_serve)

    serve.serve_model(_ModelPolicy, "127.0.0.1:5555")
    assert len(served) == 1
    assert isinstance(served[0], _ModelPolicy)
    assert served[0].loaded is True

    served.clear()
    inst = _ModelPolicy()
    serve.serve_model(inst, "127.0.0.1:5555")
    assert served == [inst]


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
            _ = framework, device, options
            self.address = address
            seen.update(env=env, address=address, tags=tags)

        def serve(self) -> None:
            seen["served"] = True

    monkeypatch.setattr(rlmesh, "EnvServer", FakeEnvServer)
    env = _Env()
    env.serve("127.0.0.1:5555", render_mode="rgb_array")
    assert env.prepared is True
    assert seen == {
        "env": ("env", True, {"render_mode": "rgb_array"}),
        "address": "127.0.0.1:5555",
        "tags": None,
        "served": True,
    }


def test_env_factory_serve_separates_serving_options_from_make_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_construct_authored_model_rejects_swallowed_binding() -> None:
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


def _tags(role: str) -> Any:
    import rlmesh.adapters as adapt

    return adapt.EnvTags(
        observation={"eef": adapt.StateTag(adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(role, dim=3)),
    )


_DELTA = _tags("action/delta_eef_pos")
_ABSOLUTE = _tags("action/eef_pos")


class _BranchEnv:
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
    branch, tags = _published(_Branched().make())
    assert branch == {"action_type": "delta"}
    assert tags == _DELTA
    assert _BranchEnv.metadata == {}


def test_branch_binds_a_positional_discriminant() -> None:
    assert _published(_Branched().make("abs"))[0] == {"action_type": "abs"}
    assert _published(_Branched().make(action_type="abs"))[1] == _ABSOLUTE


def test_branch_rejects_a_value_outside_the_declared_choices() -> None:
    from rlmesh.params import ParamError

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
            tags = _DELTA
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


class _Configured(rlmesh.Model):
    params = rlmesh.ParamSpec(rlmesh.Param("device", choices=("cpu", "cuda")))
    loads: list[dict[str, object]]

    def load(self, checkpoint: str = "default", device: str = "cpu") -> None:
        self.loads = [
            *getattr(self, "loads", []),
            {"checkpoint": checkpoint, "device": device},
        ]

    def predict(self, observation: object) -> int:
        return 0


def test_from_config_returns_the_subclass_loaded_once_with_defaults() -> None:
    model = _Configured.from_config()
    assert type(model) is _Configured
    assert model.loads == [{"checkpoint": "default", "device": "cpu"}]


def test_from_config_keywords_configure_load_exactly_once() -> None:
    model = _Configured.from_config(checkpoint="other", device="cuda")
    assert model.loads == [{"checkpoint": "other", "device": "cuda"}]


def test_from_config_validates_against_signature_and_params() -> None:
    from rlmesh.params import MissingParamError, ParamError, UnknownParamError

    with pytest.raises(UnknownParamError, match="dtype"):
        _Configured.from_config(dtype="fp16")
    with pytest.raises(ParamError, match="device"):
        _Configured.from_config(device="tpu")

    class _Required(_Configured):
        def load(self, checkpoint: str, device: str = "cpu") -> None:  # type: ignore[override]
            super().load(checkpoint=checkpoint, device=device)

    with pytest.raises(MissingParamError, match="checkpoint"):
        _Required.from_config()
    assert _Required.from_config(checkpoint="x").loads == [
        {"checkpoint": "x", "device": "cpu"}
    ]


def test_from_config_matches_the_served_binding() -> None:
    from rlmesh import serve

    local = _Configured.from_config(checkpoint="ckpt-3", device="cuda")
    served = serve._resolve_model(
        _Configured, {"checkpoint": "ckpt-3", "device": "cuda"}
    )
    assert isinstance(served, _Configured)
    assert served.loads == local.loads == [{"checkpoint": "ckpt-3", "device": "cuda"}]


def test_from_config_refuses_a_framework_base_class() -> None:
    import rlmesh.numpy

    for base in (rlmesh.Model, rlmesh.numpy.Model):
        with pytest.raises(TypeError, match="authored Model subclass"):
            base.from_config()
    assert _Configured().loads == [{"checkpoint": "default", "device": "cpu"}]
    assert rlmesh.Model(lambda obs: obs, spec=rlmesh.NO_ADAPTER) is not None
