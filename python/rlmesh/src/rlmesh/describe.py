"""Print an env/model's declared construction-parameter surface as JSON.

``python -m rlmesh.describe --env pkg:Env`` (or ``--model pkg:Model``) emits the
schema a managed dashboard reads to present, validate, and sweep variations --
out-of-band and off-GPU, so it can run in a throwaway container or a local
import. The output is forward-compatible with an OCI image label baked at build
time.

Emitted shape::

    {
        "param_spec": {...} | null,  # declared Params + extra policy
        "signature_tier": [...],  # free derived args (name/type/default)
        "variations": {...},  # enumerate_params() axes, if provided
        "catalog": [...],  # enumerate_variants() sub-envs, if provided
    }

A ``catalog`` entry is ``{"id", "params", "metadata"}`` (plus an ``"error"`` badge
when the variant's params fail off-GPU validation). The variant's ``params`` bind
only its identity dimensions; the consumer composes the free dials as the
``param_spec`` names minus those keys.
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

from ._entrypoint import resolve_entrypoint
from ._variants import Variant
from .params._resolve import describe, resolve

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve ``--env``/``--model`` and print its parameter surface as JSON."""
    parser = argparse.ArgumentParser(prog="python -m rlmesh.describe")
    parser.add_argument("--env", help="module:Class for an environment factory")
    parser.add_argument("--model", help="module:Class for a model")
    args = parser.parse_args(argv)

    if bool(args.env) == bool(args.model):
        parser.error("provide exactly one of --env or --model")

    if args.env:
        obj = resolve_entrypoint(args.env, label="env entrypoint")
        method = "make"
    else:
        obj = resolve_entrypoint(args.model, label="model entrypoint")
        method = "load"

    payload = _describe(obj, method)
    # default=repr keeps the command total: an exotic enumerate_params value
    # renders as a string rather than crashing the schema dump.
    print(json.dumps(payload, indent=2, sort_keys=True, default=repr))
    return 0


def _describe(obj: object, method: str) -> dict[str, object]:
    spec, target, enumerate_fn, catalog_fn = _resolve_target(obj, method)
    payload = describe(spec, target)
    if enumerate_fn is not None:
        try:
            payload["variations"] = _variations(enumerate_fn())
        except Exception as exc:  # describe is best-effort and off-GPU
            payload["variations_error"] = str(exc)
    if catalog_fn is not None:
        try:
            payload["catalog"] = _catalog(catalog_fn(), spec, target)
        except Exception as exc:  # a broken catalog must not crash describe
            payload["catalog_error"] = str(exc)
    return payload


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
                "enumerate_variants() must yield Variant or mapping entries; got "
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


if __name__ == "__main__":
    raise SystemExit(main())
