"""``rlmesh.describe``: the versioned, Rust-standardized metadata envelope.

Covers the pure-Python gatherer (``_gather``, ``_catalog``, ``_variations``) and
the full envelope produced through the Rust builder (``describe`` /
``describe_json`` + the classmethods).
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import rlmesh
from rlmesh._describe import _catalog, _gather, _variations

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


def test_variant_rejects_non_mapping_params() -> None:
    # dict() would silently accept a list of pairs; params must be a real mapping.
    with pytest.raises(ValueError, match="mapping"):
        rlmesh.Variant("s/0", cast("Any", [("a", 1)]))


def test_variant_eq_and_repr() -> None:
    a = rlmesh.Variant("s/0", {"a": 1}, name="Zero")
    assert a == rlmesh.Variant("s/0", {"a": 1}, name="Zero")
    assert a != rlmesh.Variant("s/1", {"a": 1}, name="Zero")
    assert a != rlmesh.Variant("s/0", {"a": 2}, name="Zero")
    assert a != rlmesh.Variant("s/0", {"a": 1}, name="One")
    assert (a == object()) is False
    assert repr(a) == "Variant(id='s/0', params={'a': 1}, metadata={'name': 'Zero'})"


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
    assert env["env_spec"]["error_type"] == "RuntimeError"
    # the rest of the envelope still ships.
    assert env["kind"] == "env" and "params" in env and "runtime" in env


class _HalfBoundedEnv:
    """CartPole-shaped elementwise bounds plus a one-sided uniform Box."""

    def __init__(self) -> None:
        import gymnasium as gym
        import numpy as np

        self.observation_space = gym.spaces.Dict(
            {
                "cart": gym.spaces.Box(
                    np.array([-4.8, -np.inf, -0.4189, -np.inf], np.float32),
                    np.array([4.8, np.inf, 0.4189, np.inf], np.float32),
                    dtype=np.float32,
                ),
                "positive": gym.spaces.Box(0.0, np.inf, (2,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)


class _HalfBoundedFactory(rlmesh.EnvFactory):
    def make(self, **kwargs: Any) -> Any:
        return _HalfBoundedEnv()


def test_partially_infinite_box_bounds_serialize_as_null() -> None:
    # A non-finite edge is JSON null ("unbounded on this edge"), not a crash:
    # json.dumps(allow_nan=False) would reject a literal Infinity.
    env = rlmesh.describe(_HalfBoundedFactory)
    spaces = env["env_spec"]["observation_space"]["details"]["spaces"]
    cart = spaces["cart"]["details"]
    assert cart["bounds_kind"] == "elementwise"
    assert cart["low"] == [-4.800000190734863, None, -0.4189000129699707, None]
    assert cart["high"] == [4.800000190734863, None, 0.4189000129699707, None]
    positive = spaces["positive"]["details"]
    assert positive["bounds_kind"] == "uniform"
    assert positive["low"] == 0.0 and positive["high"] is None
    # and the artifact is still the byte-stable Rust-normalized string.
    assert "Infinity" not in rlmesh.describe_json(_HalfBoundedFactory)


def test_model_envelope_omits_spaces() -> None:
    model = rlmesh.describe(_TinyModel)
    assert model["kind"] == "model"
    assert "env_spec" not in model and "env_tags" not in model
    # class-level read of an unset spec is null, not an error.
    assert model["model_spec"] is None


class _BatchedModel(rlmesh.Model):
    def predict_chunk_batch(self, observations: object) -> object:
        return observations


def test_model_envelope_lists_defined_corners() -> None:
    assert rlmesh.describe(_TinyModel)["corners"] == ["predict"]
    # only the authored corner, not the synthesized ones.
    assert rlmesh.describe(_BatchedModel)["corners"] == ["predict_chunk_batch"]
    assert "corners" not in rlmesh.describe(_CamArmFactory)


class _DeclaringModel(rlmesh.Model):
    native_chunk = 30

    def predict(self, observation: object) -> object:
        return 0

    def predict_chunk(self, observation: object) -> object:
        return [0]


def test_model_envelope_carries_a_declared_native_chunk() -> None:
    # K is a model property, declared at the class level and readable without
    # weights -- that is what makes it checkable before anything runs.
    assert rlmesh.describe(_DeclaringModel)["native_chunk"] == 30
    # Undeclared stays absent (no key), and it never appears on an env.
    assert "native_chunk" not in rlmesh.describe(_TinyModel)
    assert "native_chunk" not in rlmesh.describe(_CamArmFactory)


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


def test_cli_out_file_is_byte_identical_to_stdout(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # --out and stdout must emit the same bytes (both trailing-newline terminated).
    from rlmesh._describe import main

    target = f"{__name__}:_CamArmFactory"
    out = tmp_path / "envelope.json"
    assert main(["--env", target, "--out", str(out)]) == 0
    assert main(["--env", target]) == 0
    assert out.read_text(encoding="utf-8") == capsys.readouterr().out


def test_importing_the_describe_module_path_no_longer_shadows_the_function() -> None:
    # The old rlmesh/describe.py shim rebound the rlmesh.describe FUNCTION to a
    # module process-wide on import; the shim is gone (the CLI lives at
    # `python -m rlmesh._describe`), so the function always wins.
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("rlmesh.describe")
    assert callable(rlmesh.describe)


class _SpeccedModel(rlmesh.Model):
    """A model the managed probe can drive: it declares a ModelSpec."""

    @staticmethod
    def _spec() -> Any:
        import rlmesh.adapters as adapt

        return adapt.ModelSpec(
            input=adapt.Text(role=adapt.INSTRUCTION),
            output=adapt.Action(
                adapt.Actuator(adapt.ACTION_GRIPPER, dim=1, range=(-1.0, 1.0))
            ),
        )

    def predict(self, observation: object) -> int:
        return 0


_SpeccedModel.spec = _SpeccedModel._spec()


def _tags(role: str = "action/joint_pos") -> dict[str, object]:
    return {
        "observation": {
            "cam": {"type": "image", "role": "image/wrist", "part": "right_arm"}
        },
        "action": {"components": [{"role": role, "dim": 6, "part": "right_arm"}]},
    }


def _valid_describe(kind: str = "model") -> str:
    """A label the platform accepts: a specced model, or a tagged env."""
    if kind == "model":
        return rlmesh.describe_json(_SpeccedModel)
    import json

    return json.dumps({"schema_version": 1, "kind": "env", "env_tags": _tags()})


def test_check_labels_flags_the_push_blockers() -> None:
    from rlmesh._describe import DESCRIBE_LABEL, PACKAGE_LABEL, check_labels

    # No labels at all is a note, not a failure: the platform reads describe
    # off the rlmesh.serve handshake.
    report = check_labels(None)
    assert report.ok and report.failed == []
    assert any("handshake" in message for message in report.not_checked)

    # Valid describe alone passes.
    report = check_labels({DESCRIBE_LABEL: _valid_describe()})
    assert report.failed == [] and report.warnings == [], report

    # Broken JSON, wrong version, and unknown kind all fail.
    assert check_labels({DESCRIBE_LABEL: "{nope"}).failed
    assert check_labels({DESCRIBE_LABEL: '{"schema_version":2,"kind":"model"}'}).failed
    assert check_labels({DESCRIBE_LABEL: '{"schema_version":1,"kind":"thing"}'}).failed
    # No class-level spec is a warning (load() may set it; a label will not
    # carry it); an env without tags lands as not-runnable.
    report = check_labels({DESCRIBE_LABEL: rlmesh.describe_json(_TinyModel)})
    assert report.failed == [], report
    assert any("no class-level spec" in m for m in report.warnings), report
    report = check_labels({DESCRIBE_LABEL: '{"schema_version":1,"kind":"env"}'})
    assert any("declares no tags" in message for message in report.failed), report

    # Package label: bad checkpoints fail (the platform would drop them).
    labels = {
        DESCRIBE_LABEL: _valid_describe(),
        PACKAGE_LABEL: '{"schemaVersion":1,"checkpoints":[{"name":"Bad_Name","uri":"hf://x"}]}',
    }
    assert any("DNS label" in message for message in check_labels(labels).failed)
    labels[PACKAGE_LABEL] = '{"schemaVersion":1,"checkpoints":[{"name":"ok"}]}'
    assert any("no uri" in message for message in check_labels(labels).failed)


def test_check_labels_surfaces_badges_and_soft_claims_as_warnings() -> None:
    import json

    from rlmesh._describe import DESCRIBE_LABEL, PACKAGE_LABEL, check_labels

    describe = json.dumps(
        {
            "schema_version": 1,
            "kind": "env",
            "env_tags": _tags(),
            "env_spec": {"error": "sapien needs a GPU"},
            "variants": {"catalog": [{"id": "a", "error": "unbuildable"}]},
        }
    )
    package = json.dumps(
        {
            "schemaVersion": 1,
            "checkpoints": [
                {"name": "a", "uri": "hf://x", "default": True},
                {"name": "b", "uri": "hf://y", "default": True},
            ],
        }
    )
    report = check_labels({DESCRIBE_LABEL: describe, PACKAGE_LABEL: package})
    # A baked env_spec badge fails admission whatever caused it, GPU or not:
    # the label was made on the wrong machine.
    assert len(report.failed) == 1 and "env_spec" in report.failed[0], report
    assert "bake the label inside the image" in report.failed[0]
    assert not any("env_spec" in m for m in report.not_checked), report
    assert any("catalog['a']" in message for message in report.warnings)
    assert any("default checkpoints" in message for message in report.warnings)
    # A tags badge is a warning, as on the platform.
    describe = json.dumps(
        {
            "schema_version": 1,
            "kind": "env",
            "env_tags": {"error": "to_dict blew up"},
            "env_spec": {"observation_space": {}, "action_space": {}},
        }
    )
    report = check_labels({DESCRIBE_LABEL: describe})
    assert report.failed == [], report
    assert any("env_tags: to_dict blew up" in m for m in report.warnings), report


def test_check_labels_warns_about_ad_hoc_roles_but_not_blessed_or_escape() -> None:
    import json

    from rlmesh._describe import DESCRIBE_LABEL, check_labels

    def label(tags: dict[str, object]) -> dict[str, str]:
        return {
            DESCRIBE_LABEL: json.dumps(
                {"schema_version": 1, "kind": "env", "env_tags": tags}
            )
        }

    def verdict(tags: dict[str, object]) -> tuple[list[str], list[str]]:
        report = check_labels(label(tags))
        return report.failed, report.warnings

    assert verdict(_tags()) == ([], [])

    # The `x/` escape is a deliberate opt-out; it is never nudged.
    escape = {
        "observation": {"cam": {"type": "state", "role": "x/battery"}},
        "action": {"components": []},
    }
    assert verdict(escape) == ([], [])

    ad_hoc = {
        "observation": {"cam": {"type": "image", "role": "image/front"}},
        "action": {"components": []},
    }
    failures, warnings = verdict(ad_hoc)
    assert failures == []
    assert any("image/front" in message for message in warnings), warnings

    # Tags that do not resolve at all are a failure, not a nudge.
    failures, _ = verdict({"observation": {"cam": {"type": "nope"}}, "action": 3})
    assert any("does not resolve" in message for message in failures), failures


# --- runtime: the edition handshake -------------------------------------------


def test_runtime_carries_the_edition_handshake(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from rlmesh._rlmesh import build_info

    monkeypatch.delenv("RLMESH_WORKFLOW_EDITION", raising=False)
    monkeypatch.chdir(tmp_path)  # no [tool.rlmesh] in reach
    info = build_info()
    runtime = rlmesh.describe(_TinyModel)["runtime"]
    # Same names and values as the wire HandshakeRequest, from the native build.
    assert runtime["protocol_generation"] == info.protocol_generation
    assert runtime["supported_workflow_editions"] == list(
        info.supported_workflow_editions
    )
    assert runtime["supported_workflow_editions"][0] == info.workflow_edition
    # Undeclared floats to this build's newest edition, as serve declares it.
    assert runtime["preferred_workflow_edition"] == info.workflow_edition
    assert "workflow_edition_error" not in runtime

    class Declared(_TinyModel):
        workflow_edition = rlmesh.current_workflow_edition()

    runtime = rlmesh.describe(Declared)["runtime"]
    assert runtime["preferred_workflow_edition"] == rlmesh.current_workflow_edition()

    class Impossible(_TinyModel):
        workflow_edition = "1999.01"

    runtime = rlmesh.describe(Impossible)["runtime"]
    assert "1999.01" in runtime["workflow_edition_error"]
    assert runtime["preferred_workflow_edition"] == info.workflow_edition
    # The same envelope stays byte-stable across calls.
    assert rlmesh.describe_json(Declared) == rlmesh.describe_json(Declared)


# --- rlmesh check: the class-level checks ---------------------------------------


class _UntaggedFactory(rlmesh.EnvFactory):
    def make(self, **kwargs: Any) -> Any:
        return _ArmEnv()


class _CrashingFactory(rlmesh.EnvFactory):
    def make(self, **kwargs: Any) -> Any:
        raise RuntimeError("make() got an unexpected keyword argument")


def test_check_target_buckets(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    from rlmesh._describe import check_target

    monkeypatch.delenv("RLMESH_WORKFLOW_EDITION", raising=False)
    monkeypatch.chdir(tmp_path)

    report = check_target(_SpeccedModel)
    assert report.ok and report.not_checked == [], report
    assert any("model_spec" in m for m in report.passed)

    report = check_target(_TinyModel)
    assert report.ok, report
    assert any("no class-level spec" in m for m in report.warnings), report

    # The tagless env: the platform cannot adapt to it.
    report = check_target(_UntaggedFactory)
    assert any("declares no tags" in m for m in report.failed), report

    # Needs a GPU to build: not checked here, named as such.
    report = check_target(_BrokenFactory)
    assert any(
        "could not build the env on this machine" in m and "Do not bake" in m
        for m in report.not_checked
    ), report
    assert not any("env_spec" in m for m in report.failed), report

    # A construction bug is a failure.
    report = check_target(_CrashingFactory)
    assert any("env_spec: RuntimeError" in m for m in report.failed), report

    # An edition this build cannot run fails where it was declared.
    class Impossible(_SpeccedModel):
        workflow_edition = "1999.01"

    report = check_target(Impossible)
    assert any("workflow edition" in m for m in report.failed), report

    # A target that does not import fails, never raises.
    report = check_target("no_such_module_xyz:Thing")
    assert report.failed and "no_such_module_xyz" in report.failed[0]


@pytest.mark.parametrize(
    ("error_type", "message", "softened"),
    [
        ("ImportError", "No module named 'robosuite'", True),
        ("ModuleNotFoundError", "No module named 'mujoco'", True),
        ("RuntimeError", "cannot build off-GPU", True),
        ("RuntimeError", "Found no NVIDIA driver on your system", True),
        ("OutOfMemoryError", "CUDA out of memory", True),
        ("CUDARuntimeError", "cudaErrorNoDevice", True),
        ("OSError", "libEGL.so.1: cannot open shared object file", True),
        ("RuntimeError", "illegal instruction", False),
        ("TypeError", "make() got an unexpected keyword argument 'cuda'", False),
        ("KeyError", "'gpu'", False),
        ("AttributeError", "'NoneType' object has no attribute 'display'", False),
        ("FileNotFoundError", "[Errno 2] No such file or directory: 'assets'", False),
        ("ValueError", "unknown device", False),
    ],
)
def test_env_spec_badges_are_classified_by_exception_type(
    error_type: str, message: str, softened: bool
) -> None:
    from rlmesh._describe import _needs_local_resources

    assert _needs_local_resources(error_type, message) is softened


def test_cli_check_and_describe_modes(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from rlmesh._describe import DESCRIBE_LABEL, main

    specced = f"{__name__}:_SpeccedModel"
    tiny = f"{__name__}:_TinyModel"

    # --check-entrypoint: exit 1 on a failure, 0 otherwise; --json is the report.
    untagged = f"{__name__}:_UntaggedFactory"
    assert main(["--check-entrypoint", untagged, "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"failed", "warnings", "not_checked", "passed"}
    assert any("declares no tags" in m for m in report["failed"])
    assert main(["--check-entrypoint", tiny, "--json"]) == 0
    assert any(
        "no class-level spec" in m
        for m in json.loads(capsys.readouterr().out)["warnings"]
    )
    assert main(["--check-entrypoint", specced]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("ok: ") and "0 failed" in out

    # TARGET [--label]: the envelope, or the docker build --label value.
    assert main([specced]) == 0
    envelope = capsys.readouterr().out
    assert envelope == rlmesh.describe_json(specced) + "\n"
    assert main([specced, "--label"]) == 0
    assert capsys.readouterr().out == f"{DESCRIBE_LABEL}={envelope}"

    # --check-labels FILE: the label check off a JSON object.
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({DESCRIBE_LABEL: envelope.strip()}), encoding="utf-8")
    assert main(["--check-labels", str(labels), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["failed"] == []
    labels.write_text("null", encoding="utf-8")
    assert main(["--check-labels", str(labels)]) == 0
    assert "not checked: no dev.rlmesh.describe label" in capsys.readouterr().out

    # Exactly one mode.
    with pytest.raises(SystemExit):
        main([specced, "--check-entrypoint", tiny])


class _ChattyFactory(rlmesh.EnvFactory):
    """Prints while being built, the way a simulator import or make() does."""

    def make(self, **kwargs: Any) -> Any:
        print("[robosuite] banner on stdout")
        return _ArmEnv()


def test_cli_keeps_author_stdout_off_the_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from rlmesh._describe import DESCRIBE_LABEL, main

    target = f"{__name__}:_ChattyFactory"
    expected = rlmesh.describe_json(target)
    capsys.readouterr()  # the banner that direct call printed

    assert main([target, "--label"]) == 0
    out, err = capsys.readouterr()
    assert out == f"{DESCRIBE_LABEL}={expected}\n"
    assert "banner on stdout" in err

    assert main(["--check-entrypoint", target, "--json"]) == 1  # no tags
    out, err = capsys.readouterr()
    assert set(json.loads(out)) == {"failed", "warnings", "not_checked", "passed"}
    assert "banner on stdout" in err


# --- env_contracts: the contract-branch table ---------------------------------


class _BranchArmEnv:
    """Spaces that move with the branch: the 128 branch halves the camera."""

    metadata: ClassVar[dict[str, object]] = {}

    def __init__(self, width: int) -> None:
        import gymnasium as gym
        import numpy as np

        self.observation_space = gym.spaces.Dict(
            {
                "cam": gym.spaces.Box(0, 255, (width, width, 3), np.uint8),
                "eef_pos": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)


def _branch_tags(role: str) -> Any:
    import rlmesh.adapters as adapt

    return adapt.EnvTags(
        observation={"eef_pos": adapt.StateTag(adapt.EEF_POS)},
        action=adapt.Action(adapt.Actuator(role, dim=3)),
    )


_BRANCH_DELTA = _branch_tags("action/delta_eef_pos")
_BRANCH_ABS = _branch_tags("action/eef_pos")


class _BranchedFactory(rlmesh.EnvFactory):
    tags = _BRANCH_DELTA
    params = rlmesh.ParamSpec(
        rlmesh.Param("action_type", type="enum", choices=("delta", "abs")),
        rlmesh.Param("width", type="enum", choices=(256, 128)),
    )
    tag_params = ("action_type", "width")

    @classmethod
    def tags_for(cls, **params: Any) -> Any:
        return _BRANCH_DELTA if params["action_type"] == "delta" else _BRANCH_ABS

    def make(self, action_type: str = "delta", width: int = 256) -> Any:
        return _BranchArmEnv(width)


def test_unbranched_envelope_carries_no_env_contracts() -> None:
    # The no-re-probe guard: a factory with no tag_params emits exactly the
    # envelope it emitted before the field existed, byte for byte.
    assert "env_contracts" not in rlmesh.describe(_CamArmFactory)
    assert "env_contracts" not in rlmesh.describe_json(_CamArmFactory)


def test_env_contracts_lists_every_branch_default_first() -> None:
    env = rlmesh.describe(_BranchedFactory)
    contracts = env["env_contracts"]
    assert contracts["discriminants"] == ["action_type", "width"]
    branches = contracts["branches"]
    assert [b["params"] for b in branches] == [
        {"action_type": "delta", "width": 256},
        {"action_type": "delta", "width": 128},
        {"action_type": "abs", "width": 256},
        {"action_type": "abs", "width": 128},
    ]
    assert [b["default"] for b in branches] == [True, False, False, False]


def test_default_branch_is_the_top_level_contract() -> None:
    env = rlmesh.describe(_BranchedFactory)
    default = env["env_contracts"]["branches"][0]
    # Not a copy that could drift: the same objects the branch-blind view carries.
    assert default["env_tags"] == env["env_tags"]
    assert default["env_spec"] == env["env_spec"]


def test_each_branch_captures_its_own_spaces_and_tags() -> None:
    branches = rlmesh.describe(_BranchedFactory)["env_contracts"]["branches"]
    by_binding = {
        (b["params"]["action_type"], b["params"]["width"]): b for b in branches
    }
    # The discriminant binds through to make(), so the branch's spaces are real.
    spaces = by_binding[("delta", 128)]["env_spec"]["observation_space"]["details"][
        "spaces"
    ]
    assert spaces["cam"]["shape"] == [128, 128, 3]
    # ...and the contract switches with action_type.
    assert by_binding[("abs", 256)]["env_tags"]["action"]["components"][0]["role"] == (
        "action/eef_pos"
    )
    assert by_binding[("delta", 256)]["env_tags"]["action"]["components"][0][
        "role"
    ] == ("action/delta_eef_pos")


def test_check_labels_surfaces_a_non_default_branch_badge() -> None:
    import json

    from rlmesh._describe import DESCRIBE_LABEL, check_labels

    describe = json.dumps(
        {
            "schema_version": 1,
            "kind": "env",
            "env_tags": _tags(),
            "env_contracts": {
                "discriminants": ["action_type"],
                "branches": [
                    {"params": {"action_type": "delta"}, "default": True},
                    {
                        "params": {"action_type": "abs"},
                        "default": False,
                        "env_spec": {"error": "sapien needs a GPU"},
                    },
                ],
            },
        }
    )
    report = check_labels({DESCRIBE_LABEL: describe})
    assert report.failed == []
    # Named by its own binding, so the operator knows which branch is blind.
    assert any(
        "env_contracts.branches[{'action_type': 'abs'}].env_spec" in w and "sapien" in w
        for w in report.warnings
    )
