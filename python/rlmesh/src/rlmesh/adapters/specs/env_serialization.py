"""Dict round-trip for env observation features."""

from __future__ import annotations

from typing import Any

from .env_features import EnvFeature, EnvImage, EnvState, EnvText
from .serialization import (
    as_mapping,
    opt_encoding,
    opt_layout,
    opt_range,
    require_str,
)


def env_feature_to_dict(feature: EnvFeature) -> dict[str, Any]:
    """Return the JSON-compatible dict form of an observation feature."""
    if isinstance(feature, EnvImage):
        return {
            "type": "image",
            "key": feature.key,
            "role": feature.role,
            "layout": feature.layout,
            "upside_down": feature.upside_down,
        }
    if isinstance(feature, EnvState):
        return {
            "type": "state",
            "key": feature.key,
            "role": feature.role,
            "dim": feature.dim,
            "encoding": feature.encoding,
            "range": list(feature.range) if feature.range else None,
        }
    return {"type": "text", "key": feature.key, "role": feature.role}


def env_feature_from_dict(item: object) -> EnvFeature:
    """Build an observation feature from :func:`env_feature_to_dict` output."""
    data = as_mapping(item, "observation feature")
    kind = data.get("type")
    if kind == "image":
        return EnvImage(
            key=require_str(data, "key", "image feature"),
            role=require_str(data, "role", "image feature"),
            layout=opt_layout(data.get("layout"), "image feature"),
            upside_down=bool(data.get("upside_down", False)),
        )
    if kind == "state":
        dim = data.get("dim")
        return EnvState(
            key=require_str(data, "key", "state feature"),
            role=require_str(data, "role", "state feature"),
            dim=None if dim is None else int(dim),
            encoding=opt_encoding(data.get("encoding"), "state feature"),
            range=opt_range(data.get("range"), "state feature"),
        )
    if kind == "text":
        return EnvText(
            key=require_str(data, "key", "text feature"),
            role=require_str(data, "role", "text feature"),
        )
    raise ValueError(f"unknown observation feature type {kind!r}")


__all__ = ["env_feature_from_dict", "env_feature_to_dict"]
