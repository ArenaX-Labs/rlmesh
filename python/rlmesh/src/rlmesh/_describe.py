"""Describe an env/model as a versioned, Rust-standardized metadata envelope."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ._entrypoint import resolve_entrypoint
from ._rlmesh import (
    adapters_join_check,
    adapters_spec_normalize,
    describe_envelope_normalize,
)
from ._variants import Variant
from .params._resolve import describe as _describe_params
from .params._resolve import resolve

__all__ = [
    "CheckReport",
    "check_labels",
    "check_target",
    "describe",
    "describe_json",
    "main",
]


def describe(
    obj: object, *, kind: str | None = None, generated_at: str | None = None
) -> dict[str, Any]:
    """Return an env/model's full metadata envelope as a dict.

    ``obj`` may be an :class:`~rlmesh.EnvFactory`/:class:`~rlmesh.Model` class or
    instance, a bare make/predict callable, or a ``"module:Class"`` entrypoint
    string. ``kind`` (``"env"``/``"model"``) is auto-detected for a factory/model
    and required for a bare callable. ``generated_at`` is an optional RFC-3339
    timestamp (the Rust layer validates it); omit it for a content-addressable
    artifact. The returned dict is parsed from the canonical string -- use
    :func:`describe_json` when you need the exact bytes (e.g. an OCI label).
    """
    return cast(
        "dict[str, Any]",
        json.loads(describe_json(obj, kind=kind, generated_at=generated_at)),
    )


def describe_json(
    obj: object,
    *,
    kind: str | None = None,
    generated_at: str | None = None,
    served_env: object | None = None,
    workflow_edition: str | None = None,
) -> str:
    """Like :func:`describe`, but return the canonical JSON string verbatim.

    This is the byte-stable artifact (Rust-serialized); persist it as-is (no
    ``json.loads`` round-trip) when baking into OCI metadata. ``served_env`` is
    an env already built from ``obj`` (the one a server hosts): its spaces are
    read directly instead of constructing a second representative env.
    ``workflow_edition`` is the edition a server already resolved for the peer
    it hosts, which the envelope's ``runtime.workflow_edition`` must match.
    ``None`` or ``""`` (the default, for a class described on its own) resolves
    the class's declaration, else the current edition; an edition this build does not know raises :class:`ValueError`.
    """
    entrypoint: str | None = None
    if isinstance(obj, str):
        entrypoint = obj
        obj = resolve_entrypoint(obj, label="describe entrypoint")
    if _is_empty_lane_list(served_env):
        raise ValueError("served_env must hold at least one lane env, got []")
    declared = _declaring_object(obj)
    kind, _ = _kind_and_method(obj, kind)
    kind, _ = _kind_and_method(obj, kind)
    kind, method = _kind_and_method(declared, kind)
    pieces = _gather(
        declared, method, kind, entrypoint, served_env, workflow_edition, spec_of=obj
    )
    # default=repr keeps describe total; allow_nan=False matches the Rust codec's RFC 8259 strictness.
    return describe_envelope_normalize(
        kind, json.dumps(_finite(pieces), allow_nan=False, default=repr), generated_at
    )


def _finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            key: _finite(item) for key, item in cast("dict[Any, Any]", value).items()
        }
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in cast("list[Any]", value)]
    return value


def _gather(
    obj: object,
    method: str,
    kind: str,
    entrypoint: str | None,
    served_env: object | None = None,
    workflow_edition: str | None = None,
    *,
    spec_of: object | None = None,
) -> dict[str, Any]:
    spec, target, enumerate_fn, catalog_fn = _resolve_target(obj, method)
    pieces: dict[str, Any] = {
        "target": _target(obj, entrypoint),
        "params": _describe_params(spec, target),
        "runtime": {**_collect_peer_info(), **_wire_identity(obj, workflow_edition)},
    }
    variants = _variants(enumerate_fn, catalog_fn, spec, target)
    if variants:
        pieces["variants"] = variants
    if kind == "env":
        pieces["env_tags"] = _env_tags(obj)
        pieces["env_spec"] = _env_spec(
            obj, spec, target, catalog_fn, served_env=served_env
        )
        contracts = _env_contracts(obj, spec, target, catalog_fn, pieces)
        if contracts is not None:
            pieces["env_contracts"] = contracts
    else:
        pieces["model_spec"] = _model_spec(obj if spec_of is None else spec_of)
        corners = _corners(obj)
        if corners:
            pieces["corners"] = corners
        native_chunk = _declared_native_chunk(obj, corners)
        if native_chunk is not None:
            pieces["native_chunk"] = native_chunk
        built = obj if spec_of is None else spec_of
        loaded = getattr(built, "_loaded_execution_horizon", None)
        if loaded is not None:
            pieces["execution_horizon"] = loaded
    return pieces


def _declaring_object(obj: object) -> object:
    from ._models.base import ModelBase

    if not _is_model_instance(obj, ModelBase):
        return obj
    model = cast("Any", obj)
    policy: object = model._policy
    if policy is obj:
        return obj
    if policy is not None:
        return policy
    source: object = model._edition_source
    return source if source is not None else obj


def _is_model_instance(obj: object, base: type) -> bool:
    return isinstance(obj, base)


def _is_empty_lane_list(served_env: object) -> bool:
    return isinstance(served_env, list) and not cast("list[object]", served_env)


def _declared_native_chunk(obj: object, corners: Sequence[str]) -> int | None:
    """The model's static native chunk length K, read from the type."""
    cls = obj if isinstance(obj, type) or inspect.isroutine(obj) else type(obj)
    native_chunk = getattr(cls, "native_chunk", None)
    if not isinstance(native_chunk, int) or isinstance(native_chunk, bool):
        return None
    if "predict_chunk" not in corners and "predict_chunk_batch" not in corners:
        raise TypeError(
            "native_chunk declares how long ONE action chunk is, but this model "
            "defines no predict_chunk() / predict_chunk_batch(). Add a chunk "
            "corner, or drop native_chunk."
        )
    if native_chunk < 1:
        raise ValueError(
            f"native_chunk must be >= 1 (or None when undeclared), got {native_chunk!r}"
        )
    return native_chunk


def _corners(obj: object) -> list[str]:
    """The predict corners the model class actually defines."""
    from ._models.base import PREDICT_CORNERS, ModelBase

    cls: type = obj if isinstance(obj, type) else type(obj)
    base = cast("type", ModelBase)
    if issubclass(cls, base):
        return [
            name
            for name in PREDICT_CORNERS
            if getattr(cls, name, None) is not getattr(base, name)
        ]
    return [name for name in PREDICT_CORNERS if callable(getattr(cls, name, None))]


def _kind_and_method(obj: object, kind: str | None) -> tuple[str, str]:
    """Resolve (kind, construction-method), auto-detecting env vs model."""
    cls = obj if isinstance(obj, type) else type(obj)
    if kind is not None:
        if kind not in ("env", "model"):
            raise ValueError(f"kind must be 'env' or 'model', got {kind!r}")
        has_make, has_predict = hasattr(cls, "make"), hasattr(cls, "predict")
        if (kind == "env" and not has_make and has_predict) or (
            kind == "model" and not has_predict and has_make
        ):
            missing = "make" if kind == "env" else "predict"
            raise ValueError(
                f"kind={kind!r} contradicts {obj!r}, which has no {missing}()"
            )
        return kind, "make" if kind == "env" else "load"
    if hasattr(cls, "predict"):
        return "model", "load"
    if hasattr(cls, "make"):
        return "env", "make"
    raise TypeError(
        f"cannot infer kind for {obj!r}; pass kind='env' or kind='model' "
        "(a bare callable has no make/predict to classify it)"
    )


def _target(obj: object, entrypoint: str | None) -> dict[str, Any]:
    if inspect.isclass(obj):
        ref: Any = obj
    elif inspect.isroutine(obj):
        ref = obj
    else:
        ref = type(obj)
    module = getattr(ref, "__module__", None) or "?"
    name = (
        getattr(ref, "__qualname__", None)
        or getattr(ref, "__name__", None)
        or repr(ref)
    )
    return {"entrypoint": entrypoint, "qualname": f"{module}:{name}"}


def _variants(
    enumerate_fn: Callable[..., Any] | None,
    catalog_fn: Callable[..., Any] | None,
    spec: Any,
    target: Callable[..., object],
) -> dict[str, Any]:
    """Group the author's enumerate_params axes + enumerate_variants catalog."""
    out: dict[str, Any] = {}
    if enumerate_fn is not None:
        try:
            out["variations"] = _variations(enumerate_fn())
        except Exception as exc:
            out["variations_error"] = str(exc)
    if catalog_fn is not None:
        try:
            out["catalog"] = _catalog(catalog_fn(), spec, target)
        except Exception as exc:
            out["catalog_error"] = str(exc)
    return out


def _env_tags(obj: object) -> Any:
    cls = obj if isinstance(obj, type) else type(obj)
    tags = getattr(cls, "tags", None)
    return None if tags is None else _env_tags_dict(tags)


def _env_tags_dict(tags: Any) -> Any:
    try:
        return tags.to_dict()
    except Exception as exc:
        return {"error": str(exc)}


def _env_contracts(
    obj: object,
    spec: Any,
    target: Callable[..., object],
    catalog_fn: Callable[..., Any] | None,
    pieces: Mapping[str, Any],
) -> dict[str, Any] | None:
    """The factory's contract-branch table, or ``None`` for a single-contract env."""
    cls = cast("Any", obj if isinstance(obj, type) else type(obj))
    discriminants: tuple[str, ...] = getattr(cls, "tag_params", ())
    if not discriminants:
        return None
    from ._authoring import _tag_branches  # pyright: ignore[reportPrivateUsage]

    try:
        branches = _tag_branches(cls)
    except Exception as exc:
        return {"error": str(exc)}
    out: list[dict[str, Any]] = []
    for index, binding in enumerate(branches):
        default = index == 0
        out.append(
            {
                "params": dict(binding),
                "default": default,
                "env_tags": pieces["env_tags"]
                if default
                else _branch_tags(cls, binding),
                "env_spec": pieces["env_spec"]
                if default
                else _env_spec(obj, spec, target, catalog_fn, binding),
            }
        )
    return {"discriminants": list(discriminants), "branches": out}


def _branch_tags(cls: Any, binding: Mapping[str, Any]) -> Any:
    """Serialize one branch's ``tags_for`` result; badged, never fatal."""
    try:
        tags = cls.tags_for(**binding)
    except Exception as exc:
        return {"error": str(exc)}
    return None if tags is None else _env_tags_dict(tags)


def _model_spec(obj: object) -> Any:
    """Serialize the model's ``ModelSpec``; read the instance's resolved spec."""
    from .adapters.specs import ModelSpec

    spec = getattr(obj, "spec", None)
    if not isinstance(spec, ModelSpec):
        return None
    try:
        return spec.to_dict()
    except Exception as exc:
        return {"error": str(exc)}


def _env_spec(
    obj: object,
    spec: Any,
    target: Callable[..., object],
    catalog_fn: Callable[..., Any] | None,
    branch: Mapping[str, Any] | None = None,
    *,
    served_env: object | None = None,
) -> dict[str, Any]:
    """Construct one representative env and serialize its obs/action spaces."""
    env: Any
    close: Callable[[], object]
    if served_env is not None:
        env = cast("Any", served_env[0] if isinstance(served_env, list) else served_env)
        close = lambda: None  # noqa: E731
    else:
        try:
            env, close = _build_env(obj, spec, target, catalog_fn, branch)
        except Exception as exc:
            return _badge(exc)
    try:
        if _is_vector_env(env):
            return {
                "observation_space": _space_dict(env.single_observation_space),
                "action_space": _space_dict(env.single_action_space),
                "num_envs": int(env.num_envs),
            }
        return {
            "observation_space": _space_dict(env.observation_space),
            "action_space": _space_dict(env.action_space),
        }
    except Exception as exc:
        return _badge(exc)
    finally:
        close()


def _badge(exc: BaseException) -> dict[str, str]:
    """An ``env_spec`` error badge: the message plus the exception's type name."""
    return {"error": str(exc), "error_type": type(exc).__name__}


def _is_vector_env(env: object) -> bool:
    """Whether an env exposes the vector shape (``single_observation_space``)."""
    return getattr(type(env), "single_observation_space", None) is not None or (
        "single_observation_space" in getattr(env, "__dict__", {})
    )


def _build_env(
    obj: object,
    spec: Any,
    target: Callable[..., object],
    catalog_fn: Callable[..., Any] | None,
    branch: Mapping[str, Any] | None = None,
) -> tuple[Any, Callable[[], object]]:
    factory = obj() if isinstance(obj, type) else obj
    prepare = getattr(factory, "prepare", None)
    if callable(prepare):
        prepare()
    make = getattr(factory, "make", None)
    builder = make if callable(make) else factory
    env = cast("Callable[..., Any]", builder)(
        **_make_kwargs(spec, target, catalog_fn, branch)
    )
    close = getattr(env, "close", None)
    return env, (
        cast("Callable[[], object]", close) if callable(close) else (lambda: None)
    )


def _make_kwargs(
    spec: Any,
    target: Callable[..., object],
    catalog_fn: Callable[..., Any] | None,
    branch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick ``make`` kwargs: declared defaults, else the first variant's params."""
    branch = branch or {}
    try:
        return dict(resolve(spec, target, branch))
    except Exception:
        pass
    if catalog_fn is not None:
        try:
            for item in cast("Iterable[object]", catalog_fn()):
                if isinstance(item, Variant):
                    params: Mapping[str, object] = item.params
                elif isinstance(item, Mapping):
                    entry = cast("Mapping[str, object]", item)
                    params = cast("Mapping[str, object]", entry.get("params") or {})
                else:
                    continue
                return dict(resolve(spec, target, {**params, **branch}))
        except Exception:
            pass
    return dict(branch)


def _space_dict(space: object) -> dict[str, object]:
    from .spaces import Space, from_gymnasium_space
    from .spaces._internals import spec_to_json

    spec = space.spec if isinstance(space, Space) else from_gymnasium_space(space).spec
    return cast("dict[str, object]", json.loads(spec_to_json(spec)))


def _collect_peer_info() -> Mapping[str, Any]:
    from ._peer_info import collect_peer_info

    return collect_peer_info()


def _wire_identity(obj: object, resolved: str | None = None) -> dict[str, Any]:
    """The wire generation, wire revision and workflow edition a peer serving ``obj`` runs at."""
    from ._editions import resolve_workflow_edition, validate_workflow_edition
    from ._load_native import load_native

    info = load_native("build_info")()
    if resolved is not None and resolved.strip():
        edition = validate_workflow_edition(resolved)
    else:
        source = obj if isinstance(obj, type) or inspect.isroutine(obj) else type(obj)
        declared = getattr(source, "workflow_edition", None)
        edition = resolve_workflow_edition(
            declared if isinstance(declared, str) else None
        )
    return {
        "protocol_generation": str(info.protocol_generation),
        "session_wire_revision": int(load_native("SESSION_WIRE_REVISION")),
        "workflow_edition": edition,
    }


def _resolve_target(
    obj: object, method: str
) -> tuple[
    Any, Callable[..., object], Callable[..., Any] | None, Callable[..., Any] | None
]:
    """Find (param_spec, signature target, enumerate_params, enumerate_variants)."""
    if isinstance(obj, type):
        spec = getattr(obj, "params", None)
        target = _signature_target(obj, method)
        enumerate_fn = _enumerate(obj)
        catalog_fn = _enumerate_variants(obj)
    else:
        spec = getattr(type(obj), "params", None)
        target = getattr(obj, method, None)
        enumerate_fn = _enumerate(type(obj))
        catalog_fn = _enumerate_variants(type(obj))

    if not callable(target):
        return None, _as_callable(obj), None, None
    return spec, target, enumerate_fn, catalog_fn


def _signature_target(cls: type, method: str) -> Callable[..., object] | None:
    import functools

    try:
        bare = cast("object", object.__new__(cls))
    except Exception:
        bare = None
    if bare is not None:
        bound = cast("object", getattr(bare, method, None))
        if callable(bound):
            return bound
    unbound = cast("object", getattr(cls, method, None))
    if callable(unbound):
        return functools.partial(unbound, None)
    return None


def _enumerate(cls: type) -> Callable[..., Any] | None:
    fn = getattr(cls, "enumerate_params", None)
    return fn if callable(fn) else None


def _enumerate_variants(cls: type) -> Callable[..., Any] | None:
    fn = getattr(cls, "enumerate_variants", None)
    return fn if callable(fn) else None


def _catalog(
    raw: object, spec: Any, target: Callable[..., object]
) -> list[dict[str, object]]:
    """Normalize ``enumerate_variants()`` to a list of catalog entries."""
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in cast("Iterable[object]", raw):
        if isinstance(item, Variant):
            vid, params, meta = item.id, item.params, item.metadata
        elif isinstance(item, Mapping):
            entry_map = cast("Mapping[str, object]", item)
            vid = entry_map.get("id")
            params = entry_map.get("params", {})
            meta = {k: v for k, v in entry_map.items() if k not in ("id", "params")}
        else:
            raise TypeError(
                "enumerate_variants() must return (or yield) Variant or mapping entries; got "
                f"{type(item).__name__}"
            )
        if not isinstance(vid, str) or not vid:
            raise ValueError(f"variant id must be a non-empty str; got {vid!r}")
        if vid in seen:
            raise ValueError(f"duplicate variant id {vid!r}")
        seen.add(vid)
        entry: dict[str, object] = {
            "id": vid,
            "params": dict(cast("Mapping[str, object]", params)),
            "metadata": dict(meta),
        }
        try:
            # Run the same gate as a real bind but keep the author's params verbatim: resolve() fills
            # free-dial defaults, which a variant must not advertise.
            resolve(spec, target, cast("Mapping[str, object]", params))
        except Exception as exc:
            entry["error"] = str(exc)
        out.append(entry)
    return out


def _variations(raw: object) -> dict[str, list[object]]:
    """Normalize ``enumerate_params()`` to ``{axis: [values]}``."""
    if not isinstance(raw, Mapping):
        raise TypeError("enumerate_params() must return a mapping of axis -> values")
    axes = cast("Mapping[object, Sequence[object]]", raw)
    return {str(axis): _axis_values(values) for axis, values in axes.items()}


def _axis_values(values: object) -> list[object]:
    if isinstance(values, str):
        return [values]
    return list(cast("Sequence[object]", values))


def _as_callable(obj: object) -> Callable[..., object]:
    if not callable(obj):
        raise TypeError(f"cannot describe {obj!r}: not a factory, model, or callable")
    return obj


DESCRIBE_LABEL = "dev.rlmesh.describe"
PACKAGE_LABEL = "dev.rlmesh.package"

_CHECKPOINT_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

_ENTRYPOINT = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")

#: Exception types that always mean the env is broken, never that this machine lacks something.
_NEVER_SOFTENED = frozenset(
    {
        "TypeError",
        "KeyError",
        "AttributeError",
        "FileNotFoundError",
        "NameError",
        "IndexError",
        "ValueError",
        "AssertionError",
        "NotImplementedError",
        "SyntaxError",
    }
)
#: Exception types that mean a dependency is present only in the image.
_SOFTENED_TYPES = frozenset({"ImportError", "ModuleNotFoundError"})
#: Whole-word hints (so ``egl`` does not match ``illegal``) that an error came from a GPU, driver or display.
_RESOURCE_WORDS = re.compile(
    r"\b(cuda|cudnn|nvrtc|nccl|gpu|nvidia|driver|display|egl|opengl|vulkan|x11|xcb|"
    r"libgl|libcuda|no such device|shared object file)\b",
    re.IGNORECASE,
)


@dataclass
class CheckReport:
    """What a pre-push check found, in the buckets ``rlmesh check`` prints.

    ``failed``: the push would land as not-runnable, or the platform would
    reject a claim. ``warnings``: a claim the platform trims, or intent it
    cannot see. ``not_checked``: what could not be decided here (describe
    needed a GPU or assets this machine lacks, an older envelope without
    editions) and who decides it instead. ``passed`` is informational. The
    JSON form (``--json``) is ``{"failed", "warnings", "not_checked",
    "passed"}``, each a list of strings.
    """

    failed: list[str] = field(default_factory=lambda: [])
    warnings: list[str] = field(default_factory=lambda: [])
    not_checked: list[str] = field(default_factory=lambda: [])
    passed: list[str] = field(default_factory=lambda: [])

    @property
    def ok(self) -> bool:
        """Whether nothing failed (warnings and unchecked items do not block)."""
        return not self.failed

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "failed": list(self.failed),
            "warnings": list(self.warnings),
            "not_checked": list(self.not_checked),
            "passed": list(self.passed),
        }


def check_target(target: object, *, kind: str | None = None) -> CheckReport:
    """Check an env/model class before it is built.

    ``target`` is a ``"module:Class"`` entrypoint, a class, or an instance
    (see :func:`describe`). Fails on: a spec/tags that does not resolve, tags
    that do not fit the env's spaces (per contract branch), a contract branch
    without tags, a broken ``model_spec`` / ``env_tags``, an edition
    declaration this build cannot run, or a target that does not import. Warns
    on an untagged env (it runs, but only against spec-less models), on a model with no class-level ``spec`` (one set in ``load()`` serves fine
    and rides the handshake, but a baked label will not carry it;
    ``rlmesh doctor`` needs a ``ModelSpec`` to synthesize inputs without a reference env), on ad hoc roles a
    curated publish gate would refuse, and on best-effort badges elsewhere in
    the envelope (variants, a non-default contract branch's spaces). ``not_checked``
    when building the env for its spaces needed a GPU, a display, or assets
    this machine lacks.
    """
    report = CheckReport()
    try:
        envelope = describe(target, kind=kind)
    except Exception as exc:
        report.failed.append(f"describe {target!r}: {exc}")
        return report
    _check_envelope(envelope, report, "", baked=False)
    if envelope.get("kind") == "model" and _declares_no_adapter(target):
        report.warnings[:] = [
            w for w in report.warnings if not w.startswith("model_spec: ")
        ]
        report.passed.append(
            "model_spec: NO_ADAPTER (serves its own predict, with no adapter)"
        )
    return report


def _declares_no_adapter(target: object) -> bool:
    from ._models._adapter_mode import NO_ADAPTER

    if isinstance(target, str):
        try:
            target = resolve_entrypoint(target, label="check entrypoint")
        except Exception:
            return False
    return getattr(target, "spec", None) is NO_ADAPTER


def check_labels(labels: Mapping[str, str] | None) -> CheckReport:
    """Validate a built image's rlmesh labels.

    A missing ``dev.rlmesh.describe`` label is a note, not a failure:
    ``rlmesh.serve`` also exposes describe during the handshake. Run
    :func:`check_target` on the class for the class-level checks. A present
    label gets the same checks as :func:`check_target`, with stricter checks
    for a baked envelope: an ``env_spec`` / ``model_spec`` error badge fails,
    and an ``env_tags`` badge warns. A ``dev.rlmesh.package`` label is checked for well-formed
    checkpoints.
    """
    report = CheckReport()
    labels = labels or {}

    raw = labels.get(DESCRIBE_LABEL, "")
    if not raw:
        report.not_checked.append(
            f"no {DESCRIBE_LABEL} label: a runner reads describe off the "
            "rlmesh.serve handshake instead; run `rlmesh check <module:Class>` "
            "for the class-level checks"
        )
    else:
        _check_describe(raw, report)

    raw = labels.get(PACKAGE_LABEL, "")
    if raw:
        _check_package(raw, report)
    return report


def _check_describe(raw: str, report: CheckReport) -> None:
    try:
        envelope = cast("dict[str, Any]", json.loads(raw))
    except ValueError:
        report.failed.append(f"{DESCRIBE_LABEL} is not valid JSON")
        return
    if not isinstance(cast("object", envelope), dict):
        report.failed.append(f"{DESCRIBE_LABEL} must be a JSON object")
        return
    _check_envelope(envelope, report, f"{DESCRIBE_LABEL} ", baked=True)


def _check_envelope(
    envelope: Mapping[str, Any], report: CheckReport, where: str, *, baked: bool
) -> None:
    """The class-level checks, shared by a fresh envelope and a baked label."""
    if envelope.get("schema_version") != 1:
        report.failed.append(
            f"{where}schema_version {envelope.get('schema_version')!r} is not the "
            "supported version 1"
        )
    kind = envelope.get("kind")
    if kind not in ("env", "model"):
        report.failed.append(f"{where}kind {kind!r} is not 'env' or 'model'")
        return
    target = envelope.get("target")
    name = f"the {kind}"
    if isinstance(target, Mapping):
        target_map = cast("Mapping[str, object]", target)
        name = str(target_map.get("entrypoint") or target_map.get("qualname") or name)

    if kind == "model":
        spec = envelope.get("model_spec")
        if spec is None:
            report.warnings.append(
                f"{where}model_spec: {name} declares no class-level spec. Set in "
                "load()? Fine for a label-less image (the handshake carries it); "
                "a baked label will not carry it, and the managed platform cannot "
                "test a model without a ModelSpec when you onboard it"
            )
        elif isinstance(spec, Mapping) and "error" in spec:
            badge = cast("Mapping[str, object]", spec)["error"]
            report.failed.append(f"{where}model_spec: {badge}")
        else:
            report.passed.append(f"{where}model_spec: declared")
    else:
        tags = envelope.get("env_tags")
        branched = isinstance(envelope.get("env_contracts"), Mapping)
        if tags is None and branched:
            report.failed.append(
                f"{where}env_tags: {name} declares tag_params but its default "
                "branch has no tags; the managed platform rejects a contract branch "
                "without EnvTags when you onboard the image"
            )
        elif tags is None:
            report.warnings.append(
                f"{where}env_tags: {name} declares no tags; it runs against "
                "spec-less models, but rlmesh cannot adapt a spec'd model to "
                "it without EnvTags (set `tags = EnvTags(...)` on the class)"
            )
        elif isinstance(tags, Mapping) and "error" in tags:
            badge = cast("Mapping[str, object]", tags)["error"]
            report.warnings.append(f"{where}env_tags: {badge}")
        else:
            report.passed.append(f"{where}env_tags: declared")
        env_spec = envelope.get("env_spec")
        if isinstance(env_spec, Mapping) and "error" in env_spec:
            badge_map = cast("Mapping[str, object]", env_spec)
            badge = str(badge_map["error"])
            error_type = str(badge_map.get("error_type") or "")
            if baked:
                report.failed.append(
                    f"{where}env_spec: {badge} (a label that carries this badge fails "
                    "admission; bake the label inside the image, where "
                    "the env builds)"
                )
            elif _needs_local_resources(error_type, badge):
                report.not_checked.append(
                    f"{where}env_spec: could not build the env on this machine "
                    f"({error_type}: {badge}); the managed platform builds it in the "
                    "container when you onboard the image. Do not bake a describe label from this machine: it "
                    "would carry this badge and fail admission"
                )
            else:
                report.failed.append(f"{where}env_spec: {error_type}: {badge}")
        elif isinstance(env_spec, Mapping):
            report.passed.append(f"{where}env_spec: spaces read")

    for path, badge in _describe_badges(envelope):
        if path in ("env_spec", "env_tags", "model_spec"):
            continue
        report.warnings.append(f"{where}{path}: {badge}")
    _check_specs(envelope, report, where)
    _check_branches(envelope, report, where)
    _check_runtime(envelope, report, where)


def _needs_local_resources(error_type: str, message: str) -> bool:
    """Whether an ``env_spec`` badge means "not buildable here", not "broken"."""
    if error_type in _NEVER_SOFTENED:
        return False
    if error_type in _SOFTENED_TYPES or "cuda" in error_type.lower():
        return True
    return bool(_RESOURCE_WORDS.search(message)) or "/dev/nvidia" in message


def _check_specs(envelope: Mapping[str, Any], report: CheckReport, where: str) -> None:
    """A spec/tags that does not resolve fails; ad hoc roles only warn."""
    for side, key in (("env", "env_tags"), ("model", "model_spec")):
        spec = envelope.get(key)
        if not isinstance(spec, Mapping) or "error" in spec:
            continue
        raw = json.dumps(spec)
        try:
            adapters_spec_normalize(side, raw, True, "passthrough")
        except ValueError as exc:
            report.failed.append(f"{where}{key}: does not resolve: {exc}")
            continue
        if side == "env":
            _check_join(key, raw, envelope.get("env_spec"), report, where)
        try:
            adapters_spec_normalize(side, raw, True, "strict")
        except ValueError as exc:
            report.warnings.append(f"{where}{key}: {exc}")


def _check_branches(
    envelope: Mapping[str, Any], report: CheckReport, where: str
) -> None:
    contracts = envelope.get("env_contracts")
    if not isinstance(contracts, Mapping):
        return
    branches = cast("Mapping[str, object]", contracts).get("branches")
    for branch in cast("list[object]", branches if isinstance(branches, list) else []):
        if not isinstance(branch, Mapping):
            continue
        branch_map = cast("Mapping[str, Any]", branch)
        if branch_map.get("default"):
            continue
        key = f"env_contracts.branches[{branch_map.get('params')!r}].env_tags"
        tags = branch_map.get("env_tags")
        if tags is None:
            report.failed.append(
                f"{where}{key}: this contract branch has no tags; the managed "
                "platform rejects a branch without EnvTags when you onboard the image"
            )
        elif isinstance(tags, Mapping) and "error" in tags:
            badge = cast("Mapping[str, object]", tags)["error"]
            report.failed.append(
                f"{where}{key}: this contract branch's tags could not be read "
                f"({badge}); the managed platform rejects a branch without EnvTags when "
                "you onboard the image"
            )
        elif isinstance(tags, Mapping):
            _check_join(
                key, json.dumps(tags), branch_map.get("env_spec"), report, where
            )


def _check_join(
    key: str,
    tags_json: str,
    env_spec: object,
    report: CheckReport,
    where: str,
) -> None:
    """Join tags against the spaces they claim, as ``rlmesh.serve`` does at startup."""
    if not isinstance(env_spec, Mapping) or "error" in env_spec:
        return
    spaces = cast("Mapping[str, Any]", env_spec)
    try:
        advisories = adapters_join_check(
            tags_json, spaces["observation_space"], spaces["action_space"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        report.failed.append(
            f"{where}{key}: does not fit the env's spaces (rlmesh.serve refuses "
            f"it): {exc}"
        )
        return
    report.warnings.extend(f"{where}{key}: {note}" for note in advisories)


def _check_runtime(
    envelope: Mapping[str, Any], report: CheckReport, where: str
) -> None:
    raw = envelope.get("runtime")
    runtime: Mapping[str, object] = (
        cast("Mapping[str, object]", raw) if isinstance(raw, Mapping) else {}
    )
    edition = runtime.get("workflow_edition")
    if not edition:
        report.not_checked.append(
            f"{where}runtime: describe carries no workflow_edition (built with an "
            "older rlmesh); the session checks it when the image runs"
        )
        return
    try:
        from ._editions import validate_workflow_edition

        validate_workflow_edition(str(edition))
    except ValueError as exc:
        report.failed.append(f"{where}runtime: {exc}")
        return
    report.passed.append(
        f"{where}runtime: {runtime.get('protocol_generation')}, workflow edition "
        f"{edition}"
    )


def _describe_badges(envelope: Mapping[str, Any]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key in ("env_spec", "env_tags", "model_spec"):
        value = envelope.get(key)
        if isinstance(value, Mapping) and "error" in value:
            badge = cast("Mapping[str, object]", value)["error"]
            out.append((key, str(badge)))
    contracts = envelope.get("env_contracts")
    if isinstance(contracts, Mapping):
        contract_map = cast("Mapping[str, object]", contracts)
        if "error" in contract_map:
            out.append(("env_contracts", str(contract_map["error"])))
        branches = contract_map.get("branches")
        if isinstance(branches, list):
            # The default branch's badges are already reported against the top-level env_spec/env_tags;
            # only the non-default branches are added.
            for branch in cast("list[object]", branches):
                if not isinstance(branch, Mapping):
                    continue
                branch_map = cast("Mapping[str, object]", branch)
                if branch_map.get("default"):
                    continue
                for key in ("env_spec",):
                    value = branch_map.get(key)
                    if isinstance(value, Mapping) and "error" in value:
                        badge = cast("Mapping[str, object]", value)["error"]
                        out.append(
                            (
                                f"env_contracts.branches[{branch_map.get('params')!r}]"
                                f".{key}",
                                str(badge),
                            )
                        )

    variants_raw = envelope.get("variants")
    if isinstance(variants_raw, Mapping):
        variants = cast("Mapping[str, object]", variants_raw)
        for key in ("catalog_error", "variations_error"):
            if key in variants:
                out.append((f"variants.{key}", str(variants[key])))
        catalog = variants.get("catalog")
        if isinstance(catalog, list):
            for entry in cast("list[object]", catalog):
                if isinstance(entry, Mapping) and "error" in entry:
                    entry_map = cast("Mapping[str, object]", entry)
                    out.append(
                        (
                            f"variants.catalog[{entry_map.get('id')!r}]",
                            str(entry_map["error"]),
                        )
                    )
    return out


def _check_package(raw: str, report: CheckReport) -> None:
    failures, warnings = report.failed, report.warnings
    try:
        package = cast("dict[str, Any]", json.loads(raw))
    except ValueError:
        failures.append(f"{PACKAGE_LABEL} is not valid JSON")
        return
    if not isinstance(cast("object", package), dict):
        failures.append(f"{PACKAGE_LABEL} must be a JSON object")
        return
    if package.get("schemaVersion") != 1:
        failures.append(
            f"{PACKAGE_LABEL} schemaVersion {package.get('schemaVersion')!r} is "
            "not the supported version 1 (readers ignore the label)"
        )
    checkpoints = package.get("checkpoints")
    if isinstance(checkpoints, list):
        defaults = 0
        for entry in cast("list[object]", checkpoints):
            if not isinstance(entry, Mapping):
                failures.append(f"{PACKAGE_LABEL} checkpoints entries must be objects")
                continue
            entry_map = cast("Mapping[str, object]", entry)
            name = entry_map.get("name")
            if (
                not isinstance(name, str)
                or len(name) > 63
                or not _CHECKPOINT_NAME.match(name)
            ):
                failures.append(
                    f"{PACKAGE_LABEL} checkpoint name {name!r} is not a DNS label; "
                    "readers drop it"
                )
            if not entry_map.get("uri"):
                failures.append(
                    f"{PACKAGE_LABEL} checkpoint {name!r} has no uri; readers drop it"
                )
            entrypoint = entry_map.get("entrypoint")
            if entrypoint is not None and (
                not isinstance(entrypoint, str) or not _ENTRYPOINT.match(entrypoint)
            ):
                failures.append(
                    f"{PACKAGE_LABEL} checkpoint {name!r} entrypoint {entrypoint!r} "
                    "is not module:Class"
                )
            if entry_map.get("default"):
                defaults += 1
        if defaults > 1:
            warnings.append(
                f"{PACKAGE_LABEL} declares {defaults} default checkpoints; "
                "readers use the first"
            )


def _docker_labels(image: str) -> Mapping[str, str] | None:
    import subprocess

    result = subprocess.run(
        ["docker", "inspect", image, "--format", "{{json .Config.Labels}}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker inspect {image} failed: {result.stderr.strip() or 'is docker running?'}"
        )
    return cast("Mapping[str, str] | None", json.loads(result.stdout))


def _read_labels(source: str) -> Mapping[str, str] | None:
    text = sys.stdin.read() if source == "-" else open(source, encoding="utf-8").read()
    labels = cast("object", json.loads(text or "null"))
    if labels is None:
        return None
    if not isinstance(labels, dict):
        raise ValueError("labels must be a JSON object of label -> value")
    return {str(k): str(v) for k, v in cast("dict[object, object]", labels).items()}


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Route everything written to stdout, Python's and C libraries', to stderr."""
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:
        saved = None
    with contextlib.redirect_stdout(sys.stderr):
        if saved is not None:
            os.dup2(2, 1)
        try:
            yield
        finally:
            sys.stderr.flush()
            if saved is not None:
                os.dup2(saved, 1)
                os.close(saved)


def _print_report(report: CheckReport, subject: str, as_json: bool) -> int:
    if as_json:
        print(json.dumps(report.to_dict()))
        return 0 if report.ok else 1
    for message in report.failed:
        print(f"FAIL: {message}")
    for message in report.warnings:
        print(f"warn: {message}")
    for message in report.not_checked:
        print(f"not checked: {message}")
    for message in report.passed:
        print(f"ok: {message}")
    summary = (
        f"{len(report.failed)} failed, {len(report.warnings)} warnings, "
        f"{len(report.not_checked)} not checked"
    )
    print(f"{'ok' if report.ok else 'FAIL'}: {subject}: {summary}")
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Describe a class, or run the pre-push checks, from the command line.

    Modes (exactly one): ``TARGET`` / ``--env`` / ``--model`` print the
    envelope; ``--check IMAGE`` checks a built image's labels;
    ``--check-labels FILE`` checks labels read from a JSON file (``-`` for
    stdin); ``--check-entrypoint MODULE:CLASS`` checks a class. Checks exit 1
    on a failure and print ``--json`` reports for the ``rlmesh`` CLI. Whatever
    the author's code prints while being described goes to stderr; stdout
    carries only the payload.
    """
    parser = argparse.ArgumentParser(prog="python -m rlmesh._describe")
    parser.add_argument(
        "target",
        nargs="?",
        help="module:Class of an EnvFactory or Model to describe (kind detected)",
    )
    parser.add_argument("--env", help="module:Class for an environment factory")
    parser.add_argument("--model", help="module:Class for a model")
    parser.add_argument(
        "--label",
        action="store_true",
        help="print `dev.rlmesh.describe=<envelope>` for `docker build --label`; "
        "run it inside the image (the envelope's runtime block is this machine's)",
    )
    parser.add_argument(
        "--check",
        metavar="IMAGE",
        help="validate a built image's rlmesh labels before pushing",
    )
    parser.add_argument(
        "--check-labels",
        metavar="FILE",
        help="validate labels read from a JSON object file ('-' for stdin)",
    )
    parser.add_argument(
        "--check-entrypoint",
        metavar="MODULE:CLASS",
        help="check a class for packaging and contract mistakes before building",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print a check report as JSON {failed, warnings, not_checked, passed}",
    )
    parser.add_argument(
        "--out", help="write the envelope to this file instead of stdout"
    )
    parser.add_argument(
        "--generated-at",
        dest="generated_at",
        help="optional RFC-3339 timestamp to stamp (omit for a reproducible artifact)",
    )
    args = parser.parse_args(argv)

    checks = [
        value
        for value in (args.check, args.check_labels, args.check_entrypoint)
        if value
    ]
    describes = [value for value in (args.target, args.env, args.model) if value]
    if len(checks) + len(describes) != 1:
        parser.error(
            "provide exactly one of TARGET, --env, --model, --check, "
            "--check-labels, or --check-entrypoint"
        )
    if checks:
        if args.check_entrypoint:
            with _stdout_to_stderr():
                report = check_target(args.check_entrypoint)
            return _print_report(report, args.check_entrypoint, args.json)
        try:
            labels = (
                _docker_labels(args.check)
                if args.check
                else _read_labels(args.check_labels)
            )
        except (RuntimeError, ValueError, OSError) as exc:
            if args.json:
                print(json.dumps(CheckReport(failed=[str(exc)]).to_dict()))
            else:
                print(f"FAIL: {exc}")
            return 1
        return _print_report(
            check_labels(labels), args.check or args.check_labels, args.json
        )

    target = args.target or args.env or args.model
    kind = "env" if args.env else "model" if args.model else None
    with _stdout_to_stderr():
        payload = describe_json(target, kind=kind, generated_at=args.generated_at)
    if args.label:
        payload = f"{DESCRIBE_LABEL}={payload}"
        runtime = cast("dict[str, Any]", json.loads(payload[len(DESCRIBE_LABEL) + 1 :]))
        host = str(runtime.get("runtime", {}).get("os", ""))
        if host and host != "linux":
            print(
                f"RLMesh: this label was generated on {host}; a label whose "
                "runtime.os is not linux is rejected when the image is admitted. Run this inside the image "
                "(docker run --rm --entrypoint rlmesh IMAGE describe ... --label).",
                file=sys.stderr,
                flush=True,
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
