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
from typing import Any

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


def to_pair(value: Any) -> tuple[float, float] | None:
    """Convert a canonical ``[low, high]`` list to a ``(low, high)`` tuple.

    The shape readers run on Rust-validated canonical data, so this is a pure
    list->tuple conversion (tuples preserve dataclass value-equality); ``None``
    passes through.
    """
    return None if value is None else (float(value[0]), float(value[1]))
