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
    # Emit only when chunking (>1), so a non-chunked layout -- every env layout
    # and most model layouts -- serializes byte-identically to before.
    if action.execute_horizon != 1:
        out["execute_horizon"] = action.execute_horizon
    return out


def _actuator_to_dict(component: Actuator) -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": component.role,
        "dim": component.dim,
        "encoding": component.encoding,
        "range": list(component.range) if component.range else None,
        "binary": component.binary,
    }
    # scale/invert/threshold are additive env-side corrections: emit only when
    # set, so layouts that do not use them serialize byte-identically to before.
    if component.scale is not None:
        out["scale"] = component.scale
    if component.invert:
        out["invert"] = True
    if component.threshold is not None:
        out["threshold"] = component.threshold
    return out


def action_from_dict(data: Mapping[str, Any]) -> Action:
    """Build an action layout from canonical (Rust-validated) dict form."""
    components = [
        Actuator(
            role=item["role"],
            dim=int(item["dim"]),
            encoding=item.get("encoding"),
            range=to_pair(item.get("range")),
            scale=item.get("scale"),
            invert=bool(item.get("invert", False)),
            threshold=item.get("threshold"),
            binary=bool(item.get("binary", False)),
        )
        for item in data["components"]
    ]
    return Action(
        *components,
        clip=to_pair(data.get("clip")),
        execute_horizon=int(data.get("execute_horizon", 1)),
    )


__all__ = ["action_from_dict", "action_to_dict"]
