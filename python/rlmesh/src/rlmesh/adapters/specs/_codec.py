"""Route spec (de)serialization through the authoritative Rust codec.

The Rust ``rlmesh-adapters`` crate is the single source of truth for the v1
spec format. Rather than re-validate in Python (a second codec that can drift),
every spec dict is passed through :func:`normalize_spec`, which calls the Rust
serde codec to validate (frozen vocabulary, unknown-field rejection, finiteness,
dim bounds, the stack ceiling) and re-serialize the canonical form. Python keeps
only the dataclass<->dict *shape* mapping; the format authority lives in Rust.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from ..._rlmesh import adapters_spec_normalize


def normalize_spec(
    side: str, raw: Mapping[str, Any], *, allow_custom: bool
) -> dict[str, Any]:
    """Validate and canonicalize a spec dict via the Rust serde codec.

    ``side`` is ``"env"`` or ``"model"``. ``allow_custom`` is False at the
    publish boundary (rejects entrypoint custom inputs) and True for resolve /
    round-trip reads. ``allow_nan=False`` refuses the non-RFC-8259
    ``Infinity``/``NaN`` tokens the Rust codec rejects, with a clean error.
    """
    return json.loads(
        adapters_spec_normalize(side, json.dumps(raw, allow_nan=False), allow_custom)
    )


def one_or_many(value: Any) -> Any:
    """Normalize a rotation-encoding field to its canonical authored form.

    A single value (a ``str`` rotation name, a ``CustomEncoding``, or ``None``)
    passes through unchanged; a sequence of rotation names — an *accept-set*, in
    model-preference order — becomes a ``tuple`` so a frozen spec stays hashable
    and round-trips by value. A one-element sequence unwraps to its bare
    element: the Rust codec canonicalizes a one-element accept-set to the bare
    string on the wire, so unwrapping at construction keeps
    ``from_json(to_json(spec)) == spec``.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        items = tuple(cast("list[Any] | tuple[Any, ...]", value))
        if len(items) == 1:
            return items[0]
        return items
    return value


def check_accept_set(what: str, role: str | None, encoding: Any) -> None:
    """Reject a non-string accept-set member at construction.

    An accept-set is a sequence of rotation-name strings; a ``CustomEncoding``
    is a single host-side packing, never an accept-set member. Caught at the
    authoring site (naming the role) instead of surfacing later as a raw
    ``TypeError`` when the spec is JSON-serialized for resolve.
    """
    if not isinstance(encoding, tuple):
        return
    for member in cast("tuple[Any, ...]", encoding):
        if not isinstance(member, str):
            raise ValueError(
                f"{what} {role!r}: an encoding accept-set must contain only "
                f"rotation-name strings, got {type(member).__name__}; a "
                "CustomEncoding is a single encoding, not an accept-set member"
            )


def hashable_node(node: Any) -> Any:
    """Return a hashable, key-order-canonical rendering of a spec tree node.

    A Dict node is a plain ``dict``: unhashable, and the dataclass-generated
    ``__eq__`` compares it order-insensitively, so a hash must not depend on
    key order. Render it as a key-sorted tuple of ``(key, child)`` pairs;
    tuples recurse; frozen leaf dataclasses already hash by value. Applies no
    serialization, so a spec carrying a Custom input (an in-process callable)
    stays hashable.
    """
    if isinstance(node, Mapping):
        items = cast("Mapping[str, Any]", node)
        return tuple(
            (key, hashable_node(child)) for key, child in sorted(items.items())
        )
    if isinstance(node, tuple):
        return tuple(hashable_node(child) for child in cast("tuple[Any, ...]", node))
    return node


def to_pair(value: Any) -> tuple[float, float] | None:
    """Convert a canonical ``[low, high]`` list to a ``(low, high)`` tuple.

    The shape readers run on Rust-validated canonical data, so this is a pure
    list->tuple conversion (tuples preserve dataclass value-equality); ``None``
    passes through.
    """
    return None if value is None else (float(value[0]), float(value[1]))
