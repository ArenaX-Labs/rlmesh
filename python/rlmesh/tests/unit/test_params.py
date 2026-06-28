"""Declared construction parameters: resolve boundary table, describe, wiring."""

from __future__ import annotations

import warnings
from typing import cast

import pytest
import rlmesh
from rlmesh import Param, ParamSpec, Vector
from rlmesh._bootstrap.loaders import construct_authored_env, load_env_from_spec
from rlmesh.params import (
    PARAM_METADATA_KEY,
    MissingParamError,
    ParamError,
    UnknownParamError,
)
from rlmesh.params._resolve import describe, resolve, to_metadata


def _make(
    *, suite: str, task_id: int = 0, cam_width: int = 256, **kwargs: object
) -> None:
    """A stand-in target with one required, two optional, and a **kwargs tail."""


def _spec(extra: str = "forbid") -> ParamSpec:
    return ParamSpec(
        Param("suite", str, choices=("a", "b"), group="task"),
        Param("task_id", "int", default=0),
        Param("cam_width", "int", default=256),
        extra=extra,  # type: ignore[arg-type]
    )


# --- resolve: the boundary table ---------------------------------------------


def test_resolve_fills_declared_defaults_and_keeps_supplied() -> None:
    assert resolve(_spec(), _make, {"suite": "a"}) == {
        "suite": "a",
        "task_id": 0,
        "cam_width": 256,
    }


def test_resolve_coerces_integral_float_to_int() -> None:
    assert resolve(_spec(), _make, {"suite": "a", "task_id": 3.0})["task_id"] == 3


def test_resolve_rejects_bool_for_int() -> None:
    with pytest.raises(ParamError, match="expected int, got bool"):
        resolve(_spec(), _make, {"suite": "a", "task_id": True})


def test_resolve_missing_required_raises_before_construction() -> None:
    with pytest.raises(MissingParamError, match="suite"):
        resolve(_spec(), _make, {})


def test_resolve_choices_are_enforced() -> None:
    with pytest.raises(ParamError, match="not in choices"):
        resolve(_spec(), _make, {"suite": "z"})


def test_resolve_unknown_key_forbidden_by_default() -> None:
    with pytest.raises(UnknownParamError, match="robtos"):
        resolve(_spec(), _make, {"suite": "a", "robtos": 1})


def test_resolve_passthrough_forwards_rest_through_kwargs() -> None:
    out = resolve(_spec("passthrough"), _make, {"suite": "a", "horizon": 500})
    assert out["horizon"] == 500


def test_resolve_passthrough_without_kwargs_target_raises() -> None:
    def no_tail(*, suite: str) -> None: ...

    with pytest.raises(UnknownParamError, match=r"no .*kwargs"):
        resolve(
            ParamSpec(Param("suite", str), extra="passthrough"),
            no_tail,
            {"suite": "a", "extra": 1},
        )


def test_resolve_derived_signature_arg_is_type_checked() -> None:
    def target(*, gain: int = 1) -> None: ...

    # No Param for ``gain`` (signature-derived tier), but the annotation is checked.
    assert resolve(ParamSpec(), target, {"gain": 5}) == {"gain": 5}
    with pytest.raises(ParamError, match="gain"):
        resolve(ParamSpec(), target, {"gain": "nope"})


def test_resolve_none_spec_is_blind_passthrough() -> None:
    assert resolve(None, _make, {"anything": 1, "suite": "x"}) == {
        "anything": 1,
        "suite": "x",
    }


def test_resolve_warns_on_misnamed_param() -> None:
    def target(*, suite: str) -> None: ...

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve(ParamSpec(Param("typo", str, default="x")), target, {"suite": "a"})
    assert any("typo" in str(w.message) for w in caught)


# --- describe ----------------------------------------------------------------


def test_describe_separates_declared_from_signature_tier() -> None:
    def target(*, suite: str, gain: int = 1) -> None: ...

    schema = describe(ParamSpec(Param("suite", str, choices=("a",))), target)
    assert schema["param_spec"] == {
        "params": [
            {"name": "suite", "type": "str", "required": True, "choices": ["a"]}
        ],
        "extra": "forbid",
    }
    # ``gain`` is presented for free (signature tier), inferred as int from default.
    assert schema["signature_tier"] == [
        {"name": "gain", "type": "int", "default": 1, "required": False}
    ]


# --- to_metadata -------------------------------------------------------------


def test_to_metadata_marks_passthrough_tail_unvalidated() -> None:
    resolved = resolve(_spec("passthrough"), _make, {"suite": "a", "horizon": 5})
    meta = cast(
        "dict[str, dict[str, object]]",
        to_metadata(_spec("passthrough"), _make, resolved)[PARAM_METADATA_KEY],
    )
    assert meta["binding"]["horizon"] == 5
    assert meta["validated"]["suite"] is True
    assert meta["validated"]["horizon"] is False


# --- construction wiring -----------------------------------------------------


class _Env(rlmesh.EnvFactory):
    tags = None
    params = ParamSpec(
        Param("suite", str, choices=("a", "b")),
        Param("task_id", "int", default=0),
        extra="forbid",
    )

    def make(self, *, suite: str, task_id: int = 0) -> object:
        env = _StubEnv()
        env.suite = suite
        env.task_id = task_id
        return env


class _StubEnv:
    suite: str
    task_id: int


def test_construct_authored_env_validates_and_publishes_binding() -> None:
    env = construct_authored_env(_Env, suite="a", task_id=2)
    assert (env.suite, env.task_id) == ("a", 2)
    binding = env.metadata[PARAM_METADATA_KEY]["binding"]
    assert binding == {"suite": "a", "task_id": 2}


def test_construct_authored_env_rejects_bad_binding_before_make() -> None:
    with pytest.raises(UnknownParamError):
        construct_authored_env(_Env, suite="a", typo=1)


def test_load_env_from_spec_factory_kind_builds_a_binding() -> None:
    spec = {
        "kind": "factory",
        "entrypoint": f"{__name__}:_Env",
        "kwargs": {"suite": "b", "task_id": 1},
    }
    env = cast("_StubEnv", load_env_from_spec(spec))
    assert (env.suite, env.task_id) == ("b", 1)


# --- regression: validation gaps the review surfaced -------------------------


def test_resolve_rejects_non_finite_float() -> None:
    # A construction param is never legitimately NaN/inf; the float coercion
    # rejects them outright.
    spec = ParamSpec(Param("lr", "float", default=0.1))

    def target(*, lr: float = 0.1) -> None: ...

    for bad in (float("nan"), float("inf")):
        with pytest.raises(ParamError, match="finite float"):
            resolve(spec, target, {"lr": bad})


def test_resolve_validates_declared_default() -> None:
    # A typo'd default must fail before construction, not silently reach make().
    spec = ParamSpec(Param("mode", str, default="fastt", choices=("fast", "slow")))

    def target(*, mode: str = "fast") -> None: ...

    with pytest.raises(ParamError, match="choices"):
        resolve(spec, target, {})


def test_resolve_enum_choices_do_not_accept_bool_as_int() -> None:
    spec = ParamSpec(Param("level", "enum", choices=(0, 1, 2)))

    def target(*, level: object = 0) -> None: ...

    with pytest.raises(ParamError, match="choices"):
        resolve(spec, target, {"level": True})


def test_required_param_falls_back_to_signature_default() -> None:
    # Author marks a Param required but make() supplies a default; an in-process
    # path with no binding channel must use make()'s default, not crash.
    spec = ParamSpec(Param("task"))  # no Param default => required

    def target(*, task: str = "reach") -> None: ...

    assert resolve(spec, target, {}) == {}  # omitted => make() uses its own default


def test_to_metadata_preserves_structured_binding() -> None:
    # A structured (dict) binding must read back as itself, not a repr string.
    spec = ParamSpec(Param("cfg", "enum"))

    def target(*, cfg: object = None) -> None: ...

    resolved = {"cfg": {"reward_scale": 2.0}}
    meta = cast(
        "dict[str, dict[str, object]]",
        to_metadata(spec, target, resolved)[PARAM_METADATA_KEY],
    )
    assert meta["binding"]["cfg"] == {"reward_scale": 2.0}


def test_construct_authored_env_vectorizes_a_factory() -> None:
    # num_envs>1 fans make() out into a self-describing vector env so a prebuilt
    # EnvFactory image honors a SandboxVectorEnv request.
    g = pytest.importorskip("gymnasium")

    class _GymEnv(g.Env):
        observation_space = g.spaces.Box(low=0.0, high=1.0, shape=(2,))
        action_space = g.spaces.Discrete(2)

        def reset(self, *, seed: object = None, options: object = None) -> object:
            return self.observation_space.sample(), {}

        def step(self, action: object) -> object:
            return self.observation_space.sample(), 0.0, False, False, {}

    class _GymFactory(rlmesh.EnvFactory):
        tags = None
        params = None

        def make(self) -> object:
            return _GymEnv()

    vec = construct_authored_env(_GymFactory, num_envs=3)
    assert getattr(vec, "num_envs", None) == 3
    assert hasattr(vec, "single_observation_space")


# --- Vector type -------------------------------------------------------------


def _vec_target(
    *,
    cam_pos: object = (0.0, 0.0, 0.0),
    cam_quat: object = (1.0, 0.0, 0.0, 0.0),
) -> None: ...


def test_vector_accepts_tuple_and_coerces_ints_to_float() -> None:
    spec = ParamSpec(Param("cam_pos", type=Vector(3)))
    coerced = cast(
        "tuple[float, ...]",
        resolve(spec, _vec_target, {"cam_pos": (1, 2, 3)})["cam_pos"],
    )
    assert coerced == (1.0, 2.0, 3.0)
    assert all(isinstance(x, float) for x in coerced)  # proves coercion, not just ==


def test_vector_canonicalizes_json_list_to_tuple() -> None:
    # The binding path hands a JSON list; it must arrive as a tuple.
    spec = ParamSpec(Param("cam_quat", type=Vector(4)))
    out = resolve(spec, _vec_target, {"cam_quat": [1.0, 0.0, 0.0, 0.0]})
    assert out["cam_quat"] == (1.0, 0.0, 0.0, 0.0)
    assert isinstance(out["cam_quat"], tuple)


def test_vector_rejects_wrong_length() -> None:
    spec = ParamSpec(Param("cam_pos", type=Vector(3)))
    with pytest.raises(ParamError, match="3-vector"):
        resolve(spec, _vec_target, {"cam_pos": (1.0, 2.0)})


def test_vector_rejects_non_sequence() -> None:
    spec = ParamSpec(Param("cam_pos", type=Vector(3)))
    with pytest.raises(ParamError, match="3-vector"):
        resolve(spec, _vec_target, {"cam_pos": 1.0})


def test_vector_rejects_non_finite_element() -> None:
    spec = ParamSpec(Param("cam_pos", type=Vector(3)))
    with pytest.raises(ParamError, match="not finite"):
        resolve(spec, _vec_target, {"cam_pos": (1.0, float("nan"), 0.0)})


def test_vector_rejects_bool_element() -> None:
    spec = ParamSpec(Param("cam_pos", type=Vector(3)))
    with pytest.raises(ParamError, match="not a number"):
        resolve(spec, _vec_target, {"cam_pos": (True, 0.0, 0.0)})


def test_vector_unit_accepts_unit_norm_and_rejects_otherwise() -> None:
    spec = ParamSpec(Param("cam_quat", type=Vector(4, unit=True)))
    assert resolve(spec, _vec_target, {"cam_quat": (1.0, 0.0, 0.0, 0.0)})[
        "cam_quat"
    ] == (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    with pytest.raises(ParamError, match="unit-norm"):
        resolve(spec, _vec_target, {"cam_quat": (1.0, 1.0, 0.0, 0.0)})


def test_vector_choices_sweep_over_whole_vectors() -> None:
    presets = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    spec = ParamSpec(Param("cam_quat", type=Vector(4), choices=presets))
    assert resolve(spec, _vec_target, {"cam_quat": [0.0, 1.0, 0.0, 0.0]})[
        "cam_quat"
    ] == (
        0.0,
        1.0,
        0.0,
        0.0,
    )
    with pytest.raises(ParamError, match="not in choices"):
        resolve(spec, _vec_target, {"cam_quat": (0.0, 0.0, 1.0, 0.0)})


def test_vector_validates_declared_default() -> None:
    # A malformed default (wrong length) must fail before construction.
    spec = ParamSpec(Param("cam_pos", type=Vector(3), default=(0.0, 0.0)))
    with pytest.raises(ParamError, match="3-vector"):
        resolve(spec, _vec_target, {})


def test_describe_emits_vector_dim_and_unit() -> None:
    spec = ParamSpec(
        Param("cam_quat", type=Vector(4, unit=True), default=(1.0, 0.0, 0.0, 0.0))
    )
    assert describe(spec, _vec_target)["param_spec"] == {
        "params": [
            {
                "name": "cam_quat",
                "type": "vec4",
                "required": False,
                "dim": 4,
                "unit": True,
                "default": [1.0, 0.0, 0.0, 0.0],
            }
        ],
        "extra": "forbid",
    }
