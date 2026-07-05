"""Lossy metadata sanitizer mirroring the native metadata codec's acceptance.

:func:`sanitize_metadata` normalizes an env ``info``/metadata mapping into the
exact value vocabulary the native codec (``value_codec/metadata.rs`` plus its
``normalization.rs`` wrapper unwrapping) accepts, stringifying what it would
reject.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

__all__ = ["sanitize_metadata"]

_MISSING = object()


class _UnsupportedError(Exception):
    """A value the native metadata codec would reject (internal sentinel)."""


def sanitize_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce an info/metadata mapping into what RLMesh metadata accepts.

    Lossy by design: simulator handles and other rich objects (for example a
    ``sapien.Pose``) become their ``str()`` form rather than failing the serve
    path. Accepted values pass through unchanged: ``None``, ``bool``, ``int``,
    ``float``, ``str``, ``bytes``, and nested mappings/lists/tuples of them
    (tuples become lists, as in the codec). NumPy scalars/arrays and namedtuples
    are unwrapped the way the codec does (``_asdict`` / ``tolist`` / ``item``),
    enums resolve to their ``value`` with a ``name`` fallback, and non-string
    keys are stringified.

    Args:
        info: The mapping to sanitize (e.g. a Gymnasium ``info`` dict).

    Returns:
        A plain ``dict`` the native metadata codec accepts verbatim.

    Raises:
        TypeError: If ``info`` is not a mapping.
        ValueError: If ``info`` contains a reference cycle (the error names the
            key path).
    """
    unwrapped = _unwrap(info)
    if not isinstance(unwrapped, Mapping):
        raise TypeError(
            f"sanitize_metadata expects a mapping, got {type(info).__name__}"
        )
    root = cast("Mapping[Any, Any]", unwrapped)
    return {
        str(key): _sanitize(value, (str(key),), frozenset({id(root)}), strict=False)
        for key, value in root.items()
    }


def _unwrap(value: Any) -> Any:
    """Apply the codec's wrapper normalization: ``_asdict``, ``tolist``, ``item``.

    Mirrors ``normalization.rs``: the first matching method wins, so namedtuples
    become dicts and numpy/torch arrays and scalars become plain containers or
    scalars before type dispatch.
    """
    for method in ("_asdict", "tolist", "item"):
        unwrapper = getattr(value, method, None)
        if callable(unwrapper):
            return unwrapper()
    return value


def _sanitize(
    value: Any, path: tuple[str, ...], seen: frozenset[int], *, strict: bool
) -> Any:
    """Sanitize one value; ``strict`` raises :class:`_UnsupportedError` instead of str().

    The strict mode exists for the enum ``value`` probe: the codec only takes an
    enum's ``value`` when that value converts cleanly, falling back to ``name``
    otherwise, so the probe must fail rather than stringify.
    """
    value = _unwrap(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return str.__str__(value)
    if isinstance(value, bytes):
        return bytes(value)
    if isinstance(value, Mapping):
        mapping = cast("Mapping[Any, Any]", value)
        _check_cycle(mapping, path, seen)
        child_seen = seen | {id(mapping)}
        return {
            str(key): _sanitize(item, (*path, str(key)), child_seen, strict=strict)
            for key, item in mapping.items()
        }
    if isinstance(value, (list, tuple)):
        sequence = cast("list[Any] | tuple[Any, ...]", value)
        _check_cycle(sequence, path, seen)
        child_seen = seen | {id(sequence)}
        return [
            _sanitize(item, (*path, f"[{index}]"), child_seen, strict=strict)
            for index, item in enumerate(sequence)
        ]
    enum_value = getattr(value, "value", _MISSING)
    if enum_value is not _MISSING and enum_value is not value:
        try:
            return _sanitize(enum_value, path, seen, strict=True)
        except _UnsupportedError:
            pass
    enum_name = getattr(value, "name", _MISSING)
    if isinstance(enum_name, str) and enum_name is not value:
        return str.__str__(enum_name)
    if strict:
        raise _UnsupportedError(type(value).__name__)
    return str(value)


def _check_cycle(
    container: object, path: tuple[str, ...], seen: frozenset[int]
) -> None:
    """Reject a container that is its own (transitive) ancestor."""
    if id(container) in seen:
        raise ValueError(f"reference cycle in metadata at {'.'.join(path) or '<root>'}")
