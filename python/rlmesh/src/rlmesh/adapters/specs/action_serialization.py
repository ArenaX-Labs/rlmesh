"""Dict round-trip for action layouts.

The dataclass<->dict *shape* lives here; validation and canonicalization are done
by the authoritative Rust codec (see :mod:`._codec`), so the from-dict reader
operates on already-valid canonical data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._codec import to_pair
from .action import ActionComponent, ActionLayout
from .custom_encoding import CustomEncoding


def action_layout_to_dict(layout: ActionLayout) -> dict[str, Any]:
    """Return the JSON-compatible dict form of an action layout."""
    for component in layout.components:
        if isinstance(component.encoding, CustomEncoding):
            raise ValueError(
                f"action component {component.role!r} uses a CustomEncoding, "
                "whose host-side transforms cannot be serialized; resolve it "
                "locally, or add the encoding to the native vocabulary"
            )
    return {
        "components": [
            _component_to_dict(component) for component in layout.components
        ],
        "clip": list(layout.clip) if layout.clip else None,
    }


def _component_to_dict(component: ActionComponent) -> dict[str, Any]:
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


def action_layout_from_dict(data: Mapping[str, Any]) -> ActionLayout:
    """Build an action layout from canonical (Rust-validated) dict form."""
    components = [
        ActionComponent(
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
    return ActionLayout(*components, clip=to_pair(data.get("clip")))


__all__ = ["action_layout_from_dict", "action_layout_to_dict"]
