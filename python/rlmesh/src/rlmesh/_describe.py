"""Describe an env/model as a versioned, Rust-standardized metadata envelope.

``rlmesh.describe(EnvOrModel)`` (or ``python -m rlmesh._describe --env pkg:Env``)
emits the single, self-contained JSON artifact a managed service reads to present,
validate, sweep, and list an uploaded env or model. It is generated once
(build/generate-time -- constructing the env to read its spaces is allowed) and is
forward-compatible with an OCI image label baked later.

Two layers:

* **This Python module is a gatherer.** It does only the irreducibly-Python work:
  resolve the target, reflect ``make``/``load``'s signature, run the author's
  ``enumerate_*`` classmethods, and construct the env to read its obs/action
  spaces. It assembles those raw pieces into one dict.
* **Rust owns the format.** :func:`rlmesh._rlmesh.describe_envelope_normalize`
  (``rlmesh-adapters``) stamps the ``schema_version``, validates the wrapper +
  the env/model invariant, and re-serializes the whole tree through one
  ``serde_json`` pass -- so the bytes are identical across Python versions and
  any future native (C++/TS) producer that hands over the same pieces.

Emitted shape (env)::

    {
        "schema_version": 1,            # Rust-stamped
        "kind": "env",                  # or "model"
        "target": {"entrypoint", "qualname"},
        "generated_at": "...",          # only if the caller supplies one
        "env_spec": {"observation_space", "action_space"[, "num_envs"]} | {"error"},
        "env_tags": {...} | null,
        "env_contracts": {"discriminants", "branches"},   # only if tag_params
        "params": {"param_spec", "signature_tier"},
        "variants": {"catalog", "variations"[, "*_error"]},
        "runtime": {...},               # PeerInfo: python/framework versions, os, arch,
                                        # plus the edition handshake (below)
    }

``runtime`` carries what the served peer's handshake will advertise, under the
handshake's own field names: ``protocol_generation`` (the wire generation),
``supported_workflow_editions`` (every edition this build can drive, newest
first) and ``preferred_workflow_edition`` (the edition the class declares, or
this build's newest). A platform reads them to decide, before it runs the
image, whether its runtime shares an edition with it.

A model envelope drops ``env_spec``/``env_tags`` and carries ``model_spec``
plus ``corners`` (the predict corners the class defines, e.g.
``["predict_chunk_batch"]`` -- the introspected source of truth for batching
support) instead. Every gathered piece is best-effort: a failure to build
the env, read a spec, or run an enumeration becomes an ``"error"`` badge, never a
crash -- the artifact is always emitted.

The module doubles as the pre-push checker behind ``rlmesh check`` /
``rlmesh check-image`` / ``rlmesh describe``: :func:`check_target` runs the
class-level checks on a describe envelope and :func:`check_labels` the same
checks on a built image's labels, both reporting into a :class:`CheckReport`.
"""

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
from ._rlmesh import adapters_spec_normalize, describe_envelope_normalize
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
    ``workflow_edition`` is the declaration a server already resolved for the
    peer it hosts (``--workflow-edition`` and the surfaces around it), which
    the envelope's ``runtime.preferred_workflow_edition`` must match: pass the
    resolved value, or ``""`` when the server resolved none, so the envelope
    reports exactly what the handshake sends. ``None`` (the default, for a
    class described on its own) resolves the class's declaration here.
    """
    entrypoint: str | None = None
    if isinstance(obj, str):
        entrypoint = obj
        obj = resolve_entrypoint(obj, label="describe entrypoint")
    kind, method = _kind_and_method(obj, kind)
    pieces = _gather(obj, method, kind, entrypoint, served_env, workflow_edition)
    # default=repr keeps describe total: an exotic catalog/param value renders as
    # a string rather than crashing the artifact; allow_nan=False matches the Rust
    # codec's RFC-8259 strictness (NaN/Infinity are rejected, not silently passed).
    return describe_envelope_normalize(
        kind, json.dumps(_finite(pieces), allow_nan=False, default=repr), generated_at
    )


def _finite(value: object) -> object:
    """Map non-finite floats to ``None`` so the envelope stays serializable.

    A half-bounded Box (CartPole's ``low``/``high``) carries ``inf`` leaves, and
    ``allow_nan=False`` would reject them. ``null`` is serde_json's own rendering
    of a non-finite ``f64``, so the Rust normalizer round-trips it unchanged; in a
    Box edge it reads as "unbounded on this edge".
    """
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
) -> dict[str, Any]:
    """Assemble the per-language raw pieces; Rust owns the wrapper + serialization."""
    spec, target, enumerate_fn, catalog_fn = _resolve_target(obj, method)
    pieces: dict[str, Any] = {
        "target": _target(obj, entrypoint),
        "params": _describe_params(spec, target),
        "runtime": {**_collect_peer_info(), **_workflow_offer(obj, workflow_edition)},
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
        pieces["model_spec"] = _model_spec(obj)
        corners = _corners(obj)
        if corners:
            pieces["corners"] = corners
        # The model's declared native chunk length K, when it declares one at the
        # class level. A K set in load() is invisible here on purpose: describe
        # runs on the class, without weights.
        native_chunk = getattr(obj, "native_chunk", None)
        if isinstance(native_chunk, int) and not isinstance(native_chunk, bool):
            pieces["native_chunk"] = native_chunk
    return pieces


def _corners(obj: object) -> list[str]:
    """The predict corners the model class actually defines.

    Introspected, not declared, so a packaging claim like ``supportsBatching``
    can be checked against the code that ships. Identity-compared against
    ``ModelBase`` (imported lazily -- a module-level import would cycle through
    the describe classmethods); a duck-typed policy falls back to attribute
    presence.
    """
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
    """Resolve (kind, construction-method), auto-detecting env vs model.

    An explicit ``kind`` is honored (the classmethods pass it, and it is the only
    way to classify a bare callable -- preserving the old ``--env``/``--model``
    capability). Otherwise duck-type on the distinctive method: a model has
    ``predict``, a factory has ``make``. Deliberately attribute-based, not
    ``issubclass(EnvFactory/Model)`` -- importing those bases here would form an
    ``_authoring``/``_models`` <-> ``describe`` import cycle (the classmethods
    import this module).
    """
    if kind is not None:
        if kind not in ("env", "model"):
            raise ValueError(f"kind must be 'env' or 'model', got {kind!r}")
        return kind, "make" if kind == "env" else "load"
    cls = obj if isinstance(obj, type) else type(obj)
    if hasattr(cls, "predict"):
        return "model", "load"
    if hasattr(cls, "make"):
        return "env", "make"
    raise TypeError(
        f"cannot infer kind for {obj!r}; pass kind='env' or kind='model' "
        "(a bare callable has no make/predict to classify it)"
    )


def _target(obj: object, entrypoint: str | None) -> dict[str, Any]:
    """Self-identity so the artifact maps back to its source in a dashboard."""
    if inspect.isclass(obj):
        ref: Any = obj
    elif inspect.isroutine(obj):
        ref = obj  # a bare function/lambda carries its own module/qualname
    else:
        ref = type(obj)  # a factory/model instance
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
    """Group the author's enumerate_params axes + enumerate_variants catalog.

    Each is best-effort and badged on failure (running author code must not crash
    describe), kept as distinct sub-keys -- axes are independent sweep dimensions,
    the catalog is dependent named entries.
    """
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
    """Serialize the factory's ``tags`` (the obs/action contract); null/badged."""
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
    """The factory's contract-branch table, or ``None`` for a single-contract env.

    Self-describing on purpose: ``discriminants`` names the axes and every branch
    carries its *full* binding plus the ``env_tags``/``env_spec`` that binding
    produces, so a reader that never runs the author's code can say which branch a
    contract belongs to (and a static check can name the one it validated).

    The default branch (``default: true``) is not a copy: its ``env_tags`` and
    ``env_spec`` are the very objects the envelope's top-level ``env_tags`` and
    ``env_spec`` carry, so the branch-blind view and the table can never drift.
    A factory with no ``tag_params`` emits no table at all, which is what keeps
    every already-published envelope byte-identical.
    """
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
    """Serialize the model's ``ModelSpec``; read the instance's resolved spec.

    A class-level read is ``None`` for wrapped/kwarg specs, so prefer the instance
    attribute when given one. Non-``ModelSpec`` content (``NO_ADAPTER``/``None``)
    serializes as null; ``to_dict`` raising on an un-publishable custom input is
    badged, not fatal.
    """
    from .adapters.specs import ModelSpec  # lazy: avoid a heavy import at module load

    # getattr resolves the instance's spec (set at load/__init__) or the class
    # attribute -- both are the right read for their input.
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
    """Construct one representative env and serialize its obs/action spaces.

    A ``served_env`` (or the first lane of a list of them) is read as-is and left
    open; it is the caller's env, not a throwaway built here.

    One shape per *contract branch*: a factory's variants share spaces, but a
    declared ``tag_params`` discriminant may move them, so ``branch`` binds one
    discriminant combination and each branch is captured separately (see
    :func:`_env_contracts`). The whole capture is best-effort -- a
    constructor/``make`` that needs unavailable args, a missing GPU, or any
    failure becomes ``{"error":...}`` so the rest of the envelope still ships
    (e.g. a no-GPU OCI build).
    """
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
        # Vector envs expose single_* (+ num_envs), not observation_space/action_space.
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
    """An ``env_spec`` error badge: the message plus the exception's type name.

    The type is what lets a pre-build check tell "this machine cannot build
    it" (an ``ImportError`` for a simulator only the image has, a CUDA
    error) from "it is broken" (a ``TypeError`` in ``make``); see
    :func:`_needs_local_resources`.
    """
    return {"error": str(exc), "error_type": type(exc).__name__}


def _is_vector_env(env: object) -> bool:
    """Whether an env exposes the vector shape (``single_observation_space``).

    Probes the type and the instance ``__dict__`` rather than a plain ``hasattr``,
    so a gymnasium env does not trigger its deprecated wrapper-attribute
    forwarding warning just because we probed (mirrors
    :func:`rlmesh._models._connect._native_contract`).
    """
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
    """Build a representative env from a class, instance, or bare make-callable."""
    factory = obj() if isinstance(obj, type) else obj
    prepare = getattr(factory, "prepare", None)
    if callable(prepare):
        prepare()
    make = getattr(factory, "make", None)
    builder = make if callable(make) else factory  # bare make-callable
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
    """Pick ``make`` kwargs: declared defaults, else the first variant's params.

    ``branch`` pins the contract discriminants and always wins: a branch's spaces
    must be read off an env built for that branch, never off a variant that
    happens to name the same key.
    """
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
    """Serialize a space via the Rust-canonical ``spec_to_dict`` codec."""
    from .spaces import Space, from_gymnasium_space  # lazy: keep module import light
    from .spaces._internals import spec_to_dict

    spec = space.spec if isinstance(space, Space) else from_gymnasium_space(space).spec
    return spec_to_dict(spec)


def _collect_peer_info() -> Mapping[str, Any]:
    from ._peer_info import collect_peer_info  # lazy

    return collect_peer_info()


def _workflow_offer(obj: object, resolved: str | None = None) -> dict[str, Any]:
    """The edition handshake a peer serving ``obj`` will send, by its wire names.

    ``protocol_generation`` and ``supported_workflow_editions`` come from the
    native build the same way the Rust handshake builder reads them (there is
    no second list). ``preferred_workflow_edition`` is the declaration the
    server hosting ``obj`` sends: ``resolved`` when a server already settled it
    (``""`` when it settled on none), else what
    :func:`rlmesh._editions.resolve_workflow_edition` settles on for the class
    -- its ``workflow_edition`` declaration, ``RLMESH_WORKFLOW_EDITION``, or
    ``[tool.rlmesh]`` -- and in either case this build's newest edition when
    nothing is declared, exactly as the handshake spells it. A declaration
    this build cannot run is not fatal here (describe stays total); it is
    recorded as ``workflow_edition_error`` and ``rlmesh check`` fails on it.
    """
    from ._editions import resolve_workflow_edition
    from ._load_native import load_native

    info = load_native("build_info")()
    cls = obj if isinstance(obj, type) else type(obj)
    declared = getattr(cls, "workflow_edition", None)
    offer: dict[str, Any] = {
        "protocol_generation": str(info.protocol_generation),
        "supported_workflow_editions": [
            str(edition) for edition in info.supported_workflow_editions
        ],
    }
    preferred: str | None
    if resolved is not None:
        preferred = resolved.strip() or None
    else:
        try:
            preferred = resolve_workflow_edition(
                declared=declared if isinstance(declared, str) else None,
                authored=False,
            )
        except ValueError as exc:
            offer["workflow_edition_error"] = str(exc)
            preferred = None
    offer["preferred_workflow_edition"] = preferred or str(info.workflow_edition)
    return offer


def _resolve_target(
    obj: object, method: str
) -> tuple[
    Any, Callable[..., object], Callable[..., Any] | None, Callable[..., Any] | None
]:
    """Find (param_spec, signature target, enumerate_params, enumerate_variants).

    For a factory/model *class*, the construction method is bound to a bare
    instance via ``object.__new__`` so its signature reflects without running
    ``__init__`` -- describing a model must not load weights. A bare make/predict
    callable has no declared params and is described directly.
    """
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
        # A bare make-env / predict callable: no params surface, describe as-is.
        return None, _as_callable(obj), None, None
    return spec, target, enumerate_fn, catalog_fn


def _signature_target(cls: type, method: str) -> Callable[..., object] | None:
    """Return ``cls.method`` for signature reflection, without running __init__.

    Prefer a bare instance (``object.__new__``) so the bound method's signature
    drops ``self``; a class that forbids that -- a custom ``__new__`` needing args,
    or a C-extension type -- falls back to the unbound function with ``self`` bound
    off via ``partial``, so describe still emits a schema instead of crashing.
    """
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
    """Normalize ``enumerate_variants()`` to a list of catalog entries.

    Each entry is nested ``{"id", "params", "metadata"}`` -- never flattened, so an
    open metadata key cannot clobber the structural ``id``/``params`` and a future
    top-level field stays unambiguous. ``id`` must be a unique, non-empty string: a
    duplicate would silently collapse a by-id spawn map, so it is rejected here (the
    caller turns the raise into ``catalog_error``, keeping describe total). Each
    variant's ``params`` is best-effort validated against the ParamSpec + ``make``
    signature off-GPU; an unbuildable variant gets an ``"error"`` key but keeps its
    params verbatim, so the catalog never silently drops or rewrites an entry.
    """
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
            # Off-GPU buildability lint: run the same gate as a real bind, but keep
            # the author's params verbatim (resolve() fills free-dial defaults, which
            # a variant must not advertise -- it binds only identity params).
            resolve(spec, target, cast("Mapping[str, object]", params))
        except Exception as exc:
            entry["error"] = str(exc)
        out.append(entry)
    return out


def _variations(raw: object) -> dict[str, list[object]]:
    """Normalize ``enumerate_params()`` to ``{axis: [values]}``.

    Kept as independent axes -- the Cartesian product is intentionally *not*
    materialized here, so a dependent ``(suite, task)`` space never emits invalid
    combinations; the sweep planner expands what it knows is independent.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("enumerate_params() must return a mapping of axis -> values")
    axes = cast("Mapping[object, Sequence[object]]", raw)
    return {str(axis): _axis_values(values) for axis, values in axes.items()}


def _axis_values(values: object) -> list[object]:
    """Normalize one sweep axis to a list, treating a bare ``str`` as one value.

    A ``str`` satisfies ``Sequence`` but ``list("pick-place")`` explodes it into
    characters, emitting a dozen bogus single-char sweep values; guard it the same
    way :func:`rlmesh._sandbox.session.string_sequence` guards package lists.
    """
    if isinstance(values, str):
        return [values]
    return list(cast("Sequence[object]", values))


def _as_callable(obj: object) -> Callable[..., object]:
    if not callable(obj):
        raise TypeError(f"cannot describe {obj!r}: not a factory, model, or callable")
    return obj


# ---- pre-push checks (rlmesh check / check-image) --------------------------

#: OCI config labels the managed platform's probe reads off a pushed image.
DESCRIBE_LABEL = "dev.rlmesh.describe"
PACKAGE_LABEL = "dev.rlmesh.package"

#: The platform drops checkpoints whose name can't be a workload reference
#: (DNS label); mirror its pattern so the failure happens before the push.
_CHECKPOINT_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

#: Exception types that always mean the env is broken, never that this machine
#: lacks something: a ``TypeError`` in ``make`` is a bug wherever it runs, and
#: a generic ``FileNotFoundError`` is an asset the author forgot, not a driver.
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
#: Exception types that mean a dependency is present only in the image: a lazy
#: simulator import inside ``make`` cannot be satisfied on the host.
_SOFTENED_TYPES = frozenset({"ImportError", "ModuleNotFoundError"})
#: Whole-word hints (so ``egl`` does not match ``illegal``) that an error came
#: from a GPU, a driver, or a display this machine does not have.
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
    """Check an env/model class the way the platform will, before it is built.

    ``target`` is a ``"module:Class"`` entrypoint, a class, or an instance
    (see :func:`describe`). Fails on: an env without ``tags``, a spec/tags
    that does not resolve, a broken ``model_spec`` / ``env_tags``, an edition
    declaration this build cannot run, or a target that does not import. Warns
    on a model with no class-level ``spec`` (one set in ``load()`` serves fine
    and rides the handshake, but a baked label will not carry it; the managed
    probe cannot synthesize inputs without a ``ModelSpec``), on ad hoc roles a
    curated publish gate would refuse, and on best-effort badges elsewhere in
    the envelope (variants, non-default contract branches). ``not_checked``
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
    return report


def check_labels(labels: Mapping[str, str] | None) -> CheckReport:
    """Validate a built image's rlmesh labels the way the platform probe will.

    A missing ``dev.rlmesh.describe`` label is a note, not a failure: the
    platform reads describe off the ``rlmesh.serve`` handshake, so a plain
    ``python -m rlmesh.serve`` image with no labels probes fine (run
    :func:`check_target` on the class for the class-level checks). A present
    label gets the same checks as :func:`check_target` with the platform's
    verdicts for a baked envelope: an ``env_spec`` / ``model_spec`` error
    badge fails (the platform fails it, whatever caused it), an ``env_tags``
    badge warns. A ``dev.rlmesh.package`` label is checked for well-formed
    checkpoints.
    """
    report = CheckReport()
    labels = labels or {}

    raw = labels.get(DESCRIBE_LABEL, "")
    if not raw:
        report.not_checked.append(
            f"no {DESCRIBE_LABEL} label: the platform reads describe off the "
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
    """The class-level checks, shared by a fresh envelope and a baked label.

    ``where`` prefixes every message (the label name for a label check, empty
    for a class check). ``baked`` selects the platform's verdict for an
    ``env_spec`` badge: in a label it fails admission whatever caused it; on
    a class checked before the build, a badge that only says this machine
    lacks a GPU, a driver, or an in-image dependency is ``not_checked``.
    """
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
            # A class-level read: a model that sets self.spec in load() (a VLA
            # whose action dim comes from the checkpoint) serves fine and its
            # handshake envelope carries the spec, so this is not a failure --
            # but a label baked from the class will not carry it.
            report.warnings.append(
                f"{where}model_spec: {name} declares no class-level spec. Set in "
                "load()? Fine for a label-less image (the handshake carries it); "
                "a baked label will not carry it, and the managed probe cannot "
                "synthesize inputs without a ModelSpec"
            )
        elif isinstance(spec, Mapping) and "error" in spec:
            badge = cast("Mapping[str, object]", spec)["error"]
            report.failed.append(f"{where}model_spec: {badge}")
        else:
            report.passed.append(f"{where}model_spec: declared")
    else:
        tags = envelope.get("env_tags")
        if tags is None:
            report.failed.append(
                f"{where}env_tags: {name} declares no tags; the platform cannot "
                "adapt a model to it without EnvTags (set `tags = EnvTags(...)` "
                "on the class)"
            )
        elif isinstance(tags, Mapping) and "error" in tags:
            # The platform warns on a tags badge (it can still read the spaces).
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
                    f"{where}env_spec: {badge} (the platform fails a label that "
                    "carries this badge; bake the label inside the image, where "
                    "the env builds)"
                )
            elif _needs_local_resources(error_type, badge):
                report.not_checked.append(
                    f"{where}env_spec: could not build the env on this machine "
                    f"({error_type}: {badge}); the platform probe builds it in the "
                    "container. Do not bake a describe label from this machine: it "
                    "would carry this badge and fail admission"
                )
            else:
                report.failed.append(f"{where}env_spec: {error_type}: {badge}")
        elif isinstance(env_spec, Mapping):
            report.passed.append(f"{where}env_spec: spaces read")

    for path, badge in _describe_badges(envelope):
        if path in ("env_spec", "env_tags", "model_spec"):
            continue  # reported above, with their own severity
        report.warnings.append(f"{where}{path}: {badge}")
    _check_specs(envelope, report, where)
    _check_runtime(envelope, report, where)


def _needs_local_resources(error_type: str, message: str) -> bool:
    """Whether an ``env_spec`` badge means "not buildable here", not "broken".

    Decided by the exception type first (see :data:`_NEVER_SOFTENED` and
    :data:`_SOFTENED_TYPES`), then by a CUDA-named type, then by whole-word
    GPU/driver/display hints in the message. Only ``rlmesh check`` on a class
    consults this; a badge baked into a label always fails, as it does on the
    platform.
    """
    if error_type in _NEVER_SOFTENED:
        return False
    if error_type in _SOFTENED_TYPES or "cuda" in error_type.lower():
        return True
    return bool(_RESOURCE_WORDS.search(message)) or "/dev/nvidia" in message


def _check_specs(envelope: Mapping[str, Any], report: CheckReport, where: str) -> None:
    """A spec/tags that does not resolve fails; ad hoc roles only warn.

    ``passthrough`` is the structural gate every tier shares: a spec it refuses
    resolves nowhere. ``strict`` is the managed tier: registered roles pass, so
    does the ``x/`` escape namespace, and anything else is an accident waiting
    to resolve against nothing. A warning there, not a failure -- the open
    vocabulary still runs everywhere; it is the curated boundary that refuses
    it.
    """
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
        try:
            adapters_spec_normalize(side, raw, True, "strict")
        except ValueError as exc:
            report.warnings.append(f"{where}{key}: {exc}")


def _check_runtime(
    envelope: Mapping[str, Any], report: CheckReport, where: str
) -> None:
    """The edition handshake the envelope advertises (see :func:`_workflow_offer`)."""
    raw = envelope.get("runtime")
    runtime: Mapping[str, object] = (
        cast("Mapping[str, object]", raw) if isinstance(raw, Mapping) else {}
    )
    if not runtime.get("supported_workflow_editions"):
        report.not_checked.append(
            f"{where}runtime: describe carries no workflow editions (built with an "
            "older rlmesh); the runtime probe verifies the handshake"
        )
        return
    error = runtime.get("workflow_edition_error")
    if error:
        report.failed.append(f"{where}runtime: workflow edition: {error}")
        return
    report.passed.append(
        f"{where}runtime: {runtime.get('protocol_generation')}, declares "
        f"{runtime.get('preferred_workflow_edition')} of "
        f"{runtime.get('supported_workflow_editions')}"
    )


def _describe_badges(envelope: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Best-effort ``error`` badges the gatherer left in the envelope."""
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
            # The default branch's badges are already reported against the
            # top-level env_spec/env_tags (same objects), so only the non-default
            # branches -- the ones a branch-blind reader never sees -- are added,
            # each named by its own binding.
            for branch in cast("list[object]", branches):
                if not isinstance(branch, Mapping):
                    continue
                branch_map = cast("Mapping[str, object]", branch)
                if branch_map.get("default"):
                    continue
                for key in ("env_spec", "env_tags"):
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
            "not the supported version 1 (the platform will ignore the label)"
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
                    "the platform will drop it"
                )
            if not entry_map.get("uri"):
                failures.append(
                    f"{PACKAGE_LABEL} checkpoint {name!r} has no uri; "
                    "the platform will drop it"
                )
            if entry_map.get("default"):
                defaults += 1
        if defaults > 1:
            warnings.append(
                f"{PACKAGE_LABEL} declares {defaults} default checkpoints; "
                "the platform uses the first"
            )


def _docker_labels(image: str) -> Mapping[str, str] | None:
    import subprocess  # lazy: only the --check path shells out

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
    """A labels JSON object from a file, or stdin for ``-``."""
    text = sys.stdin.read() if source == "-" else open(source, encoding="utf-8").read()
    labels = cast("object", json.loads(text or "null"))
    if labels is None:
        return None
    if not isinstance(labels, dict):
        raise ValueError("labels must be a JSON object of label -> value")
    return {str(k): str(v) for k, v in cast("dict[object, object]", labels).items()}


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Route everything written to stdout, Python's and C libraries', to stderr.

    Describing runs the author's code: importing a simulator prints banners, a
    ``make()`` may print, a C library writes to fd 1 directly. The envelope
    (and the JSON report the ``rlmesh`` CLI parses) must be the only bytes on
    stdout, so fd 1 is dup2'd onto fd 2 for the duration and ``sys.stdout`` is
    pointed at ``sys.stderr`` (a replaced ``sys.stdout``, as under a test
    harness, is not backed by fd 1), then both are restored before the payload
    is written.
    """
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:
        saved = None  # no usable fd 1: the Python-level redirect still applies
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
    """Print a report (one line per finding, or JSON) and return the exit code."""
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
        help="validate a built image's rlmesh labels the way the platform probe "
        "will, before pushing",
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
                f"RLMesh: this label was generated on {host}; the platform fails a "
                "label whose runtime.os is not linux. Run this inside the image "
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
