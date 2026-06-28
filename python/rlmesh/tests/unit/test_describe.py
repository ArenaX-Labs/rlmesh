"""``rlmesh.describe``: the versioned, Rust-standardized metadata envelope.

Covers the pure-Python gatherer (``_gather``, ``_catalog``, ``_variations``) and
the full envelope produced through the Rust builder (``describe`` /
``describe_json`` + the classmethods).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import rlmesh
from rlmesh.describe import _catalog, _gather, _variations

# --- pure helpers: variations + catalog --------------------------------------


def test_variations_treats_bare_str_axis_as_one_value() -> None:
    # list("pick-place") would explode into characters; a bare str is one value.
    assert _variations({"task": "pick-place"}) == {"task": ["pick-place"]}
    assert _variations({"task": ["a", "b"]}) == {"task": ["a", "b"]}


def _noop(**_kwargs: object) -> None: ...


def test_catalog_variant_nested_shape() -> None:
    # Display info rides nested in metadata, never flattened onto the entry.
    cat = _catalog([rlmesh.Variant("s/0", {"a": 1}, name="Zero")], None, _noop)
    assert cat == [{"id": "s/0", "params": {"a": 1}, "metadata": {"name": "Zero"}}]


def test_catalog_tolerant_plain_dict_entry() -> None:
    # A plain mapping (no Variant) yields the same nested shape; non id/params keys
    # collapse into metadata.
    cat = _catalog([{"id": "s/0", "params": {"a": 1}, "name": "Zero"}], None, _noop)
    assert cat == [{"id": "s/0", "params": {"a": 1}, "metadata": {"name": "Zero"}}]


def test_catalog_rejects_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _catalog([rlmesh.Variant("x", {}), rlmesh.Variant("x", {})], None, _noop)


def test_catalog_rejects_empty_or_non_str_id() -> None:
    with pytest.raises(ValueError):
        _catalog([rlmesh.Variant("", {})], None, _noop)
    with pytest.raises(ValueError):
        _catalog([{"id": 3, "params": {}}], None, _noop)


def test_catalog_rejects_non_variant_entry() -> None:
    with pytest.raises(TypeError):
        _catalog([42], None, _noop)


def test_catalog_badges_unbuildable_variant() -> None:
    spec = rlmesh.ParamSpec(rlmesh.Param("task", type=str, choices=("a", "b")))

    def make(*, task: str = "a", n: int = 0) -> None: ...

    good = _catalog([rlmesh.Variant("ok", {"task": "a"})], spec, make)
    assert "error" not in good[0]

    bad = _catalog([rlmesh.Variant("bad", {"task": "zzz"})], spec, make)
    assert "error" in bad[0]
    # params kept verbatim despite the badge -- catalog never rewrites an entry.
    assert bad[0]["params"] == {"task": "zzz"}


def test_variant_defensively_copies_params() -> None:
    # Reusing one dict across entries in a loop must not alias the whole catalog.
    src = {"task_id": 0}
    variant = rlmesh.Variant("s/0", src)
    src["task_id"] = 99
    assert variant.params == {"task_id": 0}


# --- gatherer: grouped params / variants -------------------------------------


def test_gather_class_with_unsafe_bare_new() -> None:
    # Subclassing a C type makes object.__new__(cls) raise; the gatherer must still
    # reflect the load signature (via the partial fallback) instead of crashing.
    class _CModel(int):
        params = None

        def load(self, *, checkpoint: str = "ck") -> None: ...

        def predict(self, observation: object) -> int:
            return 0

    pieces = _gather(_CModel, "load", "model", None)
    tier = cast("list[dict[str, object]]", pieces["params"]["signature_tier"])
    assert "checkpoint" in [p["name"] for p in tier]


def test_gather_groups_catalog_under_variants() -> None:
    class _Factory:
        params = None

        @classmethod
        def enumerate_variants(cls):
            return [rlmesh.Variant("only/0", {"x": 1}, name="Only")]

        def make(self, *, x: int = 0) -> None: ...

    pieces = _gather(_Factory, "make", "env", None)
    cat = cast("list[dict[str, object]]", pieces["variants"]["catalog"])
    assert cat == [{"id": "only/0", "params": {"x": 1}, "metadata": {"name": "Only"}}]


def test_gather_omits_variants_without_enumerate() -> None:
    class _Factory:
        params = None

        def make(self, *, x: int = 0) -> None: ...

    assert "variants" not in _gather(_Factory, "make", "env", None)


def test_gather_badges_broken_catalog() -> None:
    class _Factory:
        params = None

        @classmethod
        def enumerate_variants(cls):
            raise RuntimeError("boom")

        def make(self, *, x: int = 0) -> None: ...

    pieces = _gather(_Factory, "make", "env", None)
    assert "boom" in cast("str", pieces["variants"]["catalog_error"])


# --- full envelope through the Rust builder -----------------------------------


class _ArmEnv:
    """A local env exposing gymnasium obs/action spaces (for env_spec capture)."""

    def __init__(self) -> None:
        import gymnasium as gym
        import numpy as np

        self.observation_space = gym.spaces.Dict(
            {
                "cam": gym.spaces.Box(0, 255, (8, 8, 3), np.uint8),
                "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)


class _CamArmFactory(rlmesh.EnvFactory):
    @classmethod
    def enumerate_variants(cls):
        return [rlmesh.Variant("task/0", {}, name="Only")]

    def make(self, **kwargs: Any) -> Any:
        return _ArmEnv()


class _BrokenFactory(rlmesh.EnvFactory):
    def make(self, **kwargs: Any) -> Any:
        raise RuntimeError("cannot build off-GPU")


class _TinyModel(rlmesh.Model):
    def predict(self, observation: object) -> int:
        return 0


def test_schema_constants_are_rust_owned() -> None:
    assert rlmesh.DESCRIBE_SCHEMA_VERSION == 1
    assert rlmesh.DESCRIBE_METADATA_KEY == "rlmesh.describe.v1"


def test_env_envelope_shape() -> None:
    env = rlmesh.describe(_CamArmFactory)
    assert env["schema_version"] == 1
    assert env["kind"] == "env"
    assert env["target"]["qualname"].endswith(":_CamArmFactory")
    assert env["runtime"]["language"] == "python"
    assert "param_spec" in env["params"] and "signature_tier" in env["params"]
    assert env["variants"]["catalog"][0]["id"] == "task/0"
    # happy-path spaces captured (not an error badge), and no model-only field.
    assert "error" not in env["env_spec"]
    assert "observation_space" in env["env_spec"]
    assert "model_spec" not in env
    # no wall-clock stamp unless asked.
    assert "generated_at" not in env


def test_env_spec_error_badge_keeps_envelope_total() -> None:
    env = rlmesh.describe(_BrokenFactory)
    assert "cannot build off-GPU" in env["env_spec"]["error"]
    # the rest of the envelope still ships.
    assert env["kind"] == "env" and "params" in env and "runtime" in env


def test_model_envelope_omits_spaces() -> None:
    model = rlmesh.describe(_TinyModel)
    assert model["kind"] == "model"
    assert "env_spec" not in model and "env_tags" not in model
    # class-level read of an unset spec is null, not an error.
    assert model["model_spec"] is None


def test_classmethod_matches_function() -> None:
    assert _CamArmFactory.describe() == rlmesh.describe(_CamArmFactory)
    assert _TinyModel.describe() == rlmesh.describe(_TinyModel)


def test_string_entrypoint_matches_object() -> None:
    by_string = rlmesh.describe(f"{__name__}:_CamArmFactory", kind="env")
    assert by_string["kind"] == "env"
    assert by_string["target"]["entrypoint"] == f"{__name__}:_CamArmFactory"


def test_bare_callable_requires_explicit_kind() -> None:
    with pytest.raises(TypeError, match="kind="):
        rlmesh.describe(lambda obs: 0)


def test_describe_json_is_byte_stable() -> None:
    ts = "2026-06-28T19:30:00Z"
    a = rlmesh.describe_json(_CamArmFactory, generated_at=ts)
    b = rlmesh.describe_json(_CamArmFactory, generated_at=ts)
    assert a == b
    # Rust stamps the wrapper first; nested keys are sorted by the serializer.
    assert a.startswith('{"schema_version":1,"kind":"env",')
    assert f'"generated_at":"{ts}"' in a


def test_describe_json_rejects_bad_timestamp() -> None:
    with pytest.raises(ValueError, match="RFC-3339"):
        rlmesh.describe_json(_CamArmFactory, generated_at="June 28")
