"""Describe an env/model as a versioned, Rust-standardized metadata envelope.

``rlmesh.describe(EnvOrModel)`` (or ``python -m rlmesh.describe --env pkg:Env``)
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
        "params": {"param_spec", "signature_tier"},
        "variants": {"catalog", "variations"[, "*_error"]},
        "runtime": {...},               # PeerInfo: python/framework versions, os, arch
    }

A model envelope drops ``env_spec``/``env_tags`` and carries
``model_spec`` instead. Every gathered piece is best-effort: a failure to build
the env, read a spec, or run an enumeration becomes an ``"error"`` badge, never a
crash -- the artifact is always emitted.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

from ._entrypoint import resolve_entrypoint
from ._rlmesh import describe_envelope_normalize
from ._variants import Variant
from .params._resolve import describe as _describe_params
from .params._resolve import resolve

__all__ = ["describe", "describe_json", "main"]


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
    obj: object, *, kind: str | None = None, generated_at: str | None = None
) -> str:
    """Like :func:`describe`, but return the canonical JSON string verbatim.

    This is the byte-stable artifact (Rust-serialized); persist it as-is (no
    ``json.loads`` round-trip) when baking into OCI metadata.
    """
    entrypoint: str | None = None
    if isinstance(obj, str):
        entrypoint = obj
        obj = resolve_entrypoint(obj, label="describe entrypoint")
    kind, method = _kind_and_method(obj, kind)
    pieces = _gather(obj, method, kind, entrypoint)
    # default=repr keeps describe total: an exotic catalog/param value renders as
    # a string rather than crashing the artifact; allow_nan=False matches the Rust
    # codec's RFC-8259 strictness (NaN/Infinity are rejected, not silently passed).
    return describe_envelope_normalize(
        kind, json.dumps(pieces, allow_nan=False, default=repr), generated_at
    )


def _gather(
    obj: object, method: str, kind: str, entrypoint: str | None
) -> dict[str, Any]:
    """Assemble the per-language raw pieces; Rust owns the wrapper + serialization."""
    spec, target, enumerate_fn, catalog_fn = _resolve_target(obj, method)
    pieces: dict[str, Any] = {
        "target": _target(obj, entrypoint),
        "params": _describe_params(spec, target),
        "runtime": dict(_collect_peer_info()),
    }
    variants = _variants(enumerate_fn, catalog_fn, spec, target)
    if variants:
        pieces["variants"] = variants
    if kind == "env":
        pieces["env_tags"] = _env_tags(obj)
        pieces["env_spec"] = _env_spec(obj, spec, target, catalog_fn)
    else:
        pieces["model_spec"] = _model_spec(obj)
    return pieces


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
    if tags is None:
        return None
    try:
        return tags.to_dict()
    except Exception as exc:
        return {"error": str(exc)}


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
) -> dict[str, Any]:
    """Construct one representative env and serialize its obs/action spaces.

    Single-shape by contract: an EnvFactory has one ``env_tags``, so all variants
    share spaces. The whole capture is best-effort -- a constructor/``make`` that
    needs unavailable args, a missing GPU, or any failure becomes ``{"error":...}``
    so the rest of the envelope still ships (e.g. a no-GPU OCI build).
    """
    try:
        env, close = _build_env(obj, spec, target, catalog_fn)
    except Exception as exc:
        return {"error": str(exc)}
    try:
        # Vector envs expose single_* (+ num_envs), not observation_space/action_space.
        if hasattr(env, "single_observation_space"):
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
        return {"error": str(exc)}
    finally:
        close()


def _build_env(
    obj: object,
    spec: Any,
    target: Callable[..., object],
    catalog_fn: Callable[..., Any] | None,
) -> tuple[Any, Callable[[], object]]:
    """Build a representative env from a class, instance, or bare make-callable."""
    factory = obj() if isinstance(obj, type) else obj
    prepare = getattr(factory, "prepare", None)
    if callable(prepare):
        prepare()
    make = getattr(factory, "make", None)
    builder = make if callable(make) else factory  # bare make-callable
    env = cast("Callable[..., Any]", builder)(**_make_kwargs(spec, target, catalog_fn))
    close = getattr(env, "close", None)
    return env, (
        cast("Callable[[], object]", close) if callable(close) else (lambda: None)
    )


def _make_kwargs(
    spec: Any, target: Callable[..., object], catalog_fn: Callable[..., Any] | None
) -> dict[str, Any]:
    """Pick ``make`` kwargs: declared defaults, else the first variant's params."""
    try:
        return dict(resolve(spec, target, {}))
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
                return dict(resolve(spec, target, params))
        except Exception:
            pass
    return {}


def _space_dict(space: object) -> dict[str, object]:
    """Serialize a space via the Rust-canonical ``spec_to_dict`` codec."""
    from .spaces import Space, from_gymnasium_space  # lazy: keep module import light
    from .spaces._internals import spec_to_dict

    spec = space.spec if isinstance(space, Space) else from_gymnasium_space(space).spec
    return spec_to_dict(spec)


def _collect_peer_info() -> Mapping[str, Any]:
    from ._peer_info import collect_peer_info  # lazy

    return collect_peer_info()


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


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve ``--env``/``--model`` and print (or write) the metadata envelope."""
    parser = argparse.ArgumentParser(prog="python -m rlmesh.describe")
    parser.add_argument("--env", help="module:Class for an environment factory")
    parser.add_argument("--model", help="module:Class for a model")
    parser.add_argument(
        "--out", help="write the envelope to this file instead of stdout"
    )
    parser.add_argument(
        "--generated-at",
        dest="generated_at",
        help="optional RFC-3339 timestamp to stamp (omit for a reproducible artifact)",
    )
    args = parser.parse_args(argv)

    if bool(args.env) == bool(args.model):
        parser.error("provide exactly one of --env or --model")

    target = args.env or args.model
    kind = "env" if args.env else "model"
    payload = describe_json(target, kind=kind, generated_at=args.generated_at)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
