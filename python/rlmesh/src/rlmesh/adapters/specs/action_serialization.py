"""Dict round-trip for action layouts.

The dataclass<->dict *shape* lives here; validation and canonicalization are done
by the authoritative Rust codec (see :mod:`._codec`), so the from-dict reader
operates on already-valid canonical data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._codec import to_pair
from .action import Action, Actuator
from .custom_encoding import CustomEncoding


def action_to_dict(action: Action) -> dict[str, Any]:
    """Return the JSON-compatible dict form of an action layout."""
    for component in action.components:
        if isinstance(component.encoding, CustomEncoding):
            raise ValueError(
                f"actuator {component.role!r} uses a CustomEncoding, "
                "whose host-side transforms cannot be serialized; resolve it "
                "locally, or add the encoding to the native vocabulary"
            )
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
    out["encoding"] = component.encoding
    out["range"] = list(component.range) if component.range else None
    out["binary"] = component.binary
    # scale/invert/threshold/clip/fill are additive: emit only when set, so
    # layouts that do not use them serialize byte-identically to before.
    if component.scale is not None:
        out["scale"] = component.scale
    if component.invert:
        out["invert"] = True
    if component.threshold is not None:
        out["threshold"] = component.threshold
    if component.clip:
        out["clip"] = True
    if component.fill != 0.0:
        out["fill"] = component.fill
    return out


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Build an action layout from canonical (Rust-validated) dict form."""
    components = [
        Actuator(
            role=item.get("role"),
            dim=int(item["dim"]),
            encoding=item.get("encoding"),
            range=to_pair(item.get("range")),
            scale=item.get("scale"),
            invert=bool(item.get("invert", False)),
            threshold=item.get("threshold"),
            binary=bool(item.get("binary", False)),
            clip=bool(item.get("clip", False)),
            fill=float(item.get("fill", 0.0)),
        )
        for item in data["components"]
    ]
    return Action(
        *components,
        clip=to_pair(data.get("clip")),
    )


__all__ = ["action_from_dict", "action_to_dict"]
