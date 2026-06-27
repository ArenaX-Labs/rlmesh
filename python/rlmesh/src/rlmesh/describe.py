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
        "forward_schema": {...} | null,  # best-effort Advanced tier, if forward=
        "variations": {...},  # enumerate_params() axes, if provided
    }
"""

from __future__ import annotations

import argparse
import functools
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from ._entrypoint import resolve_entrypoint
from .params import describe

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
    spec, target, enumerate_fn = _resolve_target(obj, method)
    payload = describe(spec, target)
    if enumerate_fn is not None:
        try:
            payload["variations"] = _variations(enumerate_fn())
        except Exception as exc:  # describe is best-effort and off-GPU
            payload["variations_error"] = str(exc)
    return payload


def _resolve_target(
    obj: object, method: str
) -> tuple[Any, Callable[..., object], Callable[..., Any] | None]:
    """Find the (param_spec, signature target, enumerate_params) for ``obj``.

    For a factory/model *class*, the construction method is bound to a bare
    instance via ``object.__new__`` so its signature reflects without running
    ``__init__`` -- describing a model must not load weights. A bare make/predict
    callable has no declared params and is described directly.
    """
    if isinstance(obj, type):
        spec = getattr(obj, "params", None)
        target = _signature_target(obj, method)
        enumerate_fn = _enumerate(obj)
    else:
        spec = getattr(type(obj), "params", None)
        target = getattr(obj, method, None)
        enumerate_fn = _enumerate(type(obj))

    if not callable(target):
        # A bare make-env / predict callable: no params surface, describe as-is.
        return None, _as_callable(obj), None
    return spec, target, enumerate_fn


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
