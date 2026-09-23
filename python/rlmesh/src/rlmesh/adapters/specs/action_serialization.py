"""Dict round-trip for action layouts.

The dataclass<->dict *shape* lives here; validation and canonicalization are done
by the authoritative Rust codec (see :mod:`._codec`), so the from-dict reader
operates on already-valid canonical data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ._codec import encoding_from_wire, encoding_to_wire, to_pair
from .action import Action, Actuator


def action_to_dict(action: Action) -> dict[str, Any]:
    """Return the JSON-compatible dict form of an action layout.

    A ``CustomEncoding`` actuator serializes to its ``{base, ...}`` object when
    its arms are ``module:callable`` entrypoints; an in-process callable arm is
    refused (see :func:`._codec.encoding_to_wire`).
    """
    out: dict[str, Any] = {
        "components": [_actuator_to_dict(component) for component in action.components],
        "clip": list(action.clip) if action.clip else None,
    }
    return out


def _actuator_to_dict(component: Actuator) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # A role-less (opaque) actuator omits role on the wire (mirrors the Rust
    # skip_serializing_if); a present role is emitted first as before.
    if component.role is not None:
        out["role"] = component.role
    out["dim"] = component.dim
    out["encoding"] = encoding_to_wire(component.encoding)
    out["range"] = list(component.range) if component.range else None
    out["binary"] = component.binary
    # scale/offset/invert/threshold/clip/fill are additive: emit only when set,
    # so layouts that do not use them serialize byte-identically to before. A
    # per-axis sequence goes under its own `axis_*` key; the scalar key never
    # changes type.
    for name in ("scale", "offset"):
        value = getattr(component, name)
        if isinstance(value, tuple):
            out[f"axis_{name}"] = list(cast("tuple[float, ...]", value))
        elif value is not None:
            out[name] = value
    if component.fill != 0.0:
        out["fill"] = component.fill
    if component.invert:
        out["invert"] = True
    if component.threshold is not None:
        out["threshold"] = component.threshold
    if component.clip:
        out["clip"] = True
    if component.optional:
        out["optional"] = True
    if component.frame is not None:
        out["frame"] = component.frame
    if component.reference is not None:
        out["reference"] = component.reference
    if component.part is not None:
        out["part"] = component.part
    if component.labels is not None:
        out["labels"] = list(component.labels)
    return out


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Build an action layout from canonical (Rust-validated) dict form."""
    components = [
        Actuator(
            role=item.get("role"),
            dim=int(item["dim"]),
            encoding=encoding_from_wire(item.get("encoding")),
            range=to_pair(item.get("range")),
            scale=item.get("axis_scale", item.get("scale")),
            offset=item.get("axis_offset", item.get("offset")),
            invert=bool(item.get("invert", False)),
            threshold=item.get("threshold"),
            binary=bool(item.get("binary", False)),
            clip=bool(item.get("clip", False)),
            fill=float(item.get("fill", 0.0)),
            optional=bool(item.get("optional", False)),
            frame=item.get("frame"),
            reference=item.get("reference"),
            labels=item.get("labels"),
            part=item.get("part"),
        )
        for item in data["components"]
    ]
    return Action(
        *components,
        clip=to_pair(data.get("clip")),
    )


__all__ = ["action_from_dict", "action_to_dict"]
