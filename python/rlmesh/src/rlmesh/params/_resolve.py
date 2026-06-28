"""Validate, split, and present a declared parameter surface.

Written once and reused for both bases because it needs only
``inspect.signature(target)``: an :class:`~rlmesh.EnvFactory`'s ``make`` and a
model's ``load`` are the same kind of target. Three entry points:

* :func:`resolve` -- validate incoming kwargs and apply the boundary table,
  producing the dict the constructor is called with. Runs in the container,
  before construction, so a bad binding fails before GPU cost.
* :func:`describe` -- the schema a dashboard reads to present/sweep.
* :func:`to_metadata` -- the resolved binding recorded into the contract.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from ._spec import Param, ParamSpec, Vector

#: Metadata key the resolved binding publishes under (namespaced like the
#: adapter tags ``rlmesh.adapters.v1.env_tags``). Python-only: there is no Rust
#: ``params`` consumer.
PARAM_METADATA_KEY = "rlmesh.params.v1.binding"

# JSON value types a binding/default may carry verbatim.
_JSON_SCALAR = (str, int, float, bool, type(None))

# Tolerance on the L2 norm for a ``Vector(unit=True)`` value.
_UNIT_TOL = 1e-3


class ParamError(ValueError):
    """A supplied parameter failed validation (type, choices, or range)."""


class UnknownParamError(ParamError):
    """One or more keys were neither declared nor a constructor keyword.

    Raised before construction so a typo is caught pre-GPU instead of being
    silently swallowed.
    """

    def __init__(self, names: list[str], message: str | None = None) -> None:
        self.names = names
        super().__init__(
            message
            or f"unknown parameter(s): {', '.join(names)}; declare them in the "
            "ParamSpec or set extra='passthrough'"
        )


class MissingParamError(ParamError):
    """A required declared parameter was not supplied."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"missing required parameter: {name!r}")


def resolve(
    spec: ParamSpec | None,
    target: Callable[..., object],
    kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Validate ``kwargs`` against ``spec`` + ``target``'s signature.

    Splits incoming keys into **declared** (a :class:`Param` name), **derived**
    (a keyword arg of ``target`` with no ``Param``), and **rest** (neither);
    validates declared against each ``Param`` and type-checks derived against its
    annotation; then applies the boundary table:

    * ``spec is None`` -> kwargs unchanged (blind passthrough; full back-compat).
    * rest empty -> ``{**declared, **derived}`` (with declared defaults filled).
    * rest non-empty, ``extra="forbid"`` -> :class:`UnknownParamError`.
    * rest non-empty, ``extra="passthrough"`` and ``target`` has ``**kwargs`` ->
      forward rest verbatim.
    * rest non-empty, ``extra="passthrough"`` and ``target`` has no ``**kwargs``
      -> :class:`UnknownParamError` (nowhere to forward).

    Returns the dict the constructor is called with.
    """
    if spec is None:
        return dict(kwargs)

    accepts_kwargs, sig_params = _signature_facts(target)
    declared = {p.name: p for p in spec.params}

    # Junk-drawer-drift guard: a declared Param the target cannot receive would
    # silently fall through to passthrough, so the validation it promised never
    # runs. Warn rather than fail -- the author's mistake, not the operator's.
    for name in declared:
        if name not in sig_params and not accepts_kwargs:
            warnings.warn(
                f"Param {name!r} is not a parameter of "
                f"{_target_name(target)} and is not covered by **kwargs; it "
                "will never reach the constructor",
                stacklevel=2,
            )

    out: dict[str, object] = {}
    rest: dict[str, object] = {}
    for name, value in kwargs.items():
        if name in declared:
            out[name] = _coerce(declared[name], value)
        elif name in sig_params:
            out[name] = _check_derived(name, sig_params[name].annotation, value)
        else:
            rest[name] = value

    for name in declared:
        if name in out:
            continue
        # Not supplied: the default lives in the target's signature, never on the
        # Param. A signature default => the constructor applies it (we pass
        # nothing); no signature default => the param is genuinely required.
        sig = sig_params.get(name)
        if sig is None or sig.default is inspect.Parameter.empty:
            raise MissingParamError(name)

    if rest:
        if spec.extra == "forbid":
            raise UnknownParamError(sorted(rest))
        if not accepts_kwargs:
            raise UnknownParamError(
                sorted(rest),
                message=(
                    f"extra='passthrough' but {_target_name(target)} has no "
                    f"**kwargs to forward: {', '.join(sorted(rest))}"
                ),
            )
        out.update(rest)
    return out


def describe(
    spec: ParamSpec | None, target: Callable[..., object]
) -> dict[str, object]:
    """Return the schema a dashboard reads to present and sweep.

    Pure, off-GPU, and importable: declared params and the free signature-derived
    tier. The dependent ``variations`` axis is filled by the caller from
    ``enumerate_params()`` (see :mod:`rlmesh.describe`).
    """
    _, sig_params = _signature_facts(target)
    declared = {p.name for p in spec.params} if spec else set[str]()
    signature_tier = [
        {
            "name": name,
            "type": _annotation_name(param.annotation, param.default),
            "default": _json_safe(_param_default(param)),
            "required": param.default is inspect.Parameter.empty,
        }
        for name, param in sig_params.items()
        if name not in declared
    ]
    return {
        "param_spec": _spec_to_dict(spec, sig_params),
        "signature_tier": signature_tier,
    }


def to_metadata(
    spec: ParamSpec | None,
    target: Callable[..., object],
    resolved: Mapping[str, object],
) -> dict[str, object]:
    """Return a metadata fragment recording the resolved binding.

    Carries the full binding (so the operator sees exactly what was sent, even
    undeclared params) plus a per-key ``validated`` flag -- a confidence badge in
    the UI, ``False`` on the forwarded passthrough tail.
    """
    _, sig_params = _signature_facts(target)
    declared = {p.name for p in spec.params} if spec else set[str]()
    return {
        PARAM_METADATA_KEY: {
            "binding": {k: _json_safe(v) for k, v in resolved.items()},
            "param_spec": _spec_to_dict(spec, sig_params),
            "validated": {
                name: (name in declared or name in sig_params) for name in resolved
            },
        }
    }


# --- internals ---------------------------------------------------------------


def _signature_facts(
    target: Callable[..., object],
) -> tuple[bool, dict[str, inspect.Parameter]]:
    """Return ``(accepts_**kwargs, {keyword_name: Parameter})`` for ``target``.

    Un-introspectable targets (some C builtins) report ``(True, {})`` -- accept
    anything, know no names -- matching the graceful fallback the gym loader uses.
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return True, {}
    accepts_kwargs = False
    params: dict[str, inspect.Parameter] = {}
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
        elif param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            params[param.name] = param
    return accepts_kwargs, params


_TYPE_NAMES: dict[type, str] = {int: "int", float: "float", str: "str", bool: "bool"}


def _type_name(declared: type | str | Vector) -> str:
    """Normalize a ``Param.type`` to its canonical string name."""
    if isinstance(declared, Vector):
        return f"vec{declared.dim}"
    if isinstance(declared, str):
        return declared
    return _TYPE_NAMES.get(declared, getattr(declared, "__name__", "str"))


def _coerce(param: Param, value: object) -> object:
    """Validate and lightly coerce ``value`` against ``param``."""
    if isinstance(param.type, Vector):
        coerced = _coerce_vector(param.name, param.type, value)
    else:
        coerced = _coerce_scalar(param.name, _type_name(param.type), value)
    if param.choices is not None and not _in_choices(coerced, param.choices):
        raise ParamError(
            f"{param.name}: {coerced!r} not in choices {list(param.choices)}"
        )
    return coerced


def _coerce_vector(name: str, spec: Vector, value: object) -> tuple[float, ...]:
    """Validate a fixed-length float vector; canonicalize list/tuple -> tuple.

    The binding path matters: a vector bound via JSON (env var, recorded
    metadata) arrives as a ``list``, so accept both and always return a tuple.
    """
    if not isinstance(value, (list, tuple)):
        raise ParamError(
            f"{name}: expected a {spec.dim}-vector, got {_typename(value)}"
        )
    elements = cast("Sequence[object]", value)
    if len(elements) != spec.dim:
        raise ParamError(
            f"{name}: expected a {spec.dim}-vector, got length {len(elements)}"
        )
    out: list[float] = []
    for element in elements:
        # ``bool`` is an ``int`` subclass; reject it so ``True`` is not silently 1.
        if isinstance(element, bool) or not isinstance(element, (int, float)):
            raise ParamError(f"{name}: vector element is not a number: {element!r}")
        result = float(element)
        if not math.isfinite(result):
            raise ParamError(f"{name}: vector element is not finite: {element!r}")
        out.append(result)
    if spec.unit:
        norm = math.sqrt(sum(x * x for x in out))
        if abs(norm - 1.0) > _UNIT_TOL:
            raise ParamError(
                f"{name}: expected a unit-norm vector, got norm {norm:.6f}"
            )
    return tuple(out)


def _coerce_scalar(name: str, kind: str, value: object) -> object:
    """Type-check + coerce one scalar; ``enum``/unknown kinds pass through."""
    if kind == "bool":
        if not isinstance(value, bool):
            raise ParamError(f"{name}: expected bool, got {_typename(value)}")
        return value
    if kind == "int":
        # ``bool`` is an ``int`` subclass; reject it so ``True`` is not silently 1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamError(f"{name}: expected int, got {_typename(value)}")
        if isinstance(value, float):
            if not value.is_integer():
                raise ParamError(f"{name}: expected int, got non-integral {value!r}")
            return int(value)
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ParamError(f"{name}: expected float, got {_typename(value)}")
        result = float(value)
        # A construction param is never legitimately NaN/inf; reject them outright.
        if not math.isfinite(result):
            raise ParamError(f"{name}: expected a finite float, got {result!r}")
        return result
    if kind == "str":
        if not isinstance(value, str):
            raise ParamError(f"{name}: expected str, got {_typename(value)}")
        return value
    # "enum" (domain is the choices set) or an opaque custom type: no coercion.
    return value


def _in_choices(value: object, choices: tuple[object, ...]) -> bool:
    """Membership test that does not conflate ``bool`` with ``int``.

    ``True == 1`` and ``False == 0`` in Python, so a plain ``in`` check lets a bool
    slip through an int-valued choices set (and an int through a bool set). Require
    the bool-ness of the value and the matched choice to agree -- the same int/bool
    guard the int coercion branch applies, extended to the enum/custom path.
    """
    for choice in choices:
        if value == choice and isinstance(value, bool) == isinstance(choice, bool):
            return True
    return False


def _check_derived(name: str, annotation: object, value: object) -> object:
    """Best-effort type-check a signature-derived arg against its annotation.

    Only the four scalar annotations are checked; anything else (unannotated, or a
    complex/structured annotation) passes through verbatim. Handles a stringized
    annotation too (``from __future__ import annotations`` makes ``x: int`` read
    back as the string ``"int"``), which this codebase uses throughout.
    """
    if isinstance(annotation, type):
        kind = _TYPE_NAMES.get(annotation)
    elif isinstance(annotation, str) and annotation in _TYPE_NAMES.values():
        kind = annotation
    else:
        kind = None
    if kind is None:
        return value
    return _coerce_scalar(name, kind, value)


def _spec_to_dict(
    spec: ParamSpec | None,
    sig_params: Mapping[str, inspect.Parameter] | None = None,
) -> dict[str, object] | None:
    if spec is None:
        return None
    return {
        "params": [_param_to_dict(p, sig_params) for p in spec.params],
        "extra": spec.extra,
    }


def _param_to_dict(
    param: Param,
    sig_params: Mapping[str, inspect.Parameter] | None = None,
) -> dict[str, object]:
    # The signature is the single source of the default: a param with a signature
    # default is optional and presents it; one without is required.
    sig = sig_params.get(param.name) if sig_params is not None else None
    has_default = sig is not None and sig.default is not inspect.Parameter.empty
    out: dict[str, object] = {
        "name": param.name,
        "type": _type_name(param.type),
        "required": not has_default,
    }
    if isinstance(param.type, Vector):
        out["dim"] = param.type.dim
        if param.type.unit:
            out["unit"] = True
    if sig is not None and sig.default is not inspect.Parameter.empty:
        out["default"] = _json_safe(sig.default)
    if param.choices is not None:
        out["choices"] = [_json_safe(c) for c in param.choices]
    if param.description:
        out["description"] = param.description
    if param.group is not None:
        out["group"] = param.group
    return out


def _annotation_name(annotation: object, default: object) -> str:
    """Name a signature param's type: annotation first, else inferred from default.

    ``inspect.Parameter.empty`` is itself a class, so the "no annotation" case
    must be tested by identity before the ``isinstance(_, type)`` branch.
    """
    if annotation is not inspect.Parameter.empty:
        if isinstance(annotation, type):
            return _TYPE_NAMES.get(annotation, getattr(annotation, "__name__", "str"))
        return str(annotation)
    if default is not inspect.Parameter.empty and default is not None:
        return _TYPE_NAMES.get(type(default), "str")
    return "str"


def _param_default(param: inspect.Parameter) -> object:
    return None if param.default is inspect.Parameter.empty else param.default


def _json_safe(value: object) -> object:
    """Keep JSON scalars verbatim; render anything else as its ``repr``.

    A describe/metadata payload is serialized to JSON; a forwarded default may be
    a live object (a robosuite config), so fall back to a string rather than
    crash the schema.
    """
    if isinstance(value, _JSON_SCALAR):
        return value
    # Preserve JSON-shaped structures so a structured binding (a dict from
    # RLMESH_MAKE_KWARGS) reads back as itself, not as a mangled repr string; a
    # live non-JSON object still falls through to repr below.
    if isinstance(value, Mapping):
        items = cast("Mapping[object, object]", value)
        return {str(k): _json_safe(v) for k, v in items.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in cast("Sequence[object]", value)]
    return repr(value)


def _typename(value: object) -> str:
    return type(value).__name__


def _target_name(target: Callable[..., object]) -> str:
    return getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))


__all__ = [
    "PARAM_METADATA_KEY",
    "MissingParamError",
    "ParamError",
    "UnknownParamError",
    "describe",
    "resolve",
    "to_metadata",
]
