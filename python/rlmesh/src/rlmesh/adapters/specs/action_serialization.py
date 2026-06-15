"""Dict round-trip for action layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .action import ActionComponent, ActionLayout
from .custom_encoding import CustomEncoding
from .serialization import (
    as_mapping,
    opt_encoding,
    opt_range,
    require_sequence,
    require_str,
)


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
    """Build an action layout from :func:`action_layout_to_dict` output."""
    components: list[ActionComponent] = []
    for entry in require_sequence(data, "components"):
        item = as_mapping(entry, "action component")
        raw_dim = item.get("dim", 0)
        components.append(
            ActionComponent(
                role=require_str(item, "role", "action component"),
                # A null dim becomes 0 (rejected downstream) rather than a raw
                # TypeError from int(None).
                dim=int(raw_dim) if raw_dim is not None else 0,
                encoding=opt_encoding(item.get("encoding"), "action component"),
                range=opt_range(item.get("range"), "action component"),
                scale=_opt_float(item.get("scale")),
                invert=bool(item.get("invert", False)),
                threshold=_opt_float(item.get("threshold")),
                binary=bool(item.get("binary", False)),
            )
        )
    return ActionLayout(
        *components,
        clip=opt_range(data.get("clip"), "action layout"),
    )


def _opt_float(value: Any) -> float | None:
    return float(value) if value is not None else None


__all__ = ["action_layout_from_dict", "action_layout_to_dict"]
