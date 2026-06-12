"""Validation primitives shared by the spec (de)serializers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .layouts import IMAGE_LAYOUTS, ImageLayout
from .rotations import ROTATION_DIMS, RotationEncoding


def load_json_mapping(payload: str) -> Mapping[str, Any]:
    """Parse a JSON string that must hold an object at the top level."""
    data: object = json.loads(payload)
    if not isinstance(data, Mapping):
        raise ValueError("expected a JSON object at the top level")
    return cast(Mapping[str, Any], data)


def as_mapping(value: object, label: str) -> Mapping[str, Any]:
    """Validate that a value is a mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Fetch a field that must be a mapping."""
    return as_mapping(data.get(key), f"spec field {key!r}")


def require_sequence(data: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Fetch a field that must be a non-string sequence."""
    value = data.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"spec field {key!r} must be a sequence")
    return cast(Sequence[Any], value)


def require_str(data: Mapping[str, Any], key: str, label: str) -> str:
    """Fetch a field that must be a string."""
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{label} field {key!r} must be a string")
    return value


def opt_range(value: object, label: str) -> tuple[float, float] | None:
    """Parse an optional ``(low, high)`` pair."""
    if value is None:
        return None
    if not isinstance(value, Sequence):
        raise ValueError(f"{label} range must be a (low, high) pair")
    pair = cast(Sequence[Any], value)
    if len(pair) != 2:
        raise ValueError(f"{label} range must be a (low, high) pair")
    return (float(pair[0]), float(pair[1]))


def opt_encoding(value: object, label: str) -> RotationEncoding | None:
    """Parse an optional rotation encoding name."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in ROTATION_DIMS:
        raise ValueError(
            f"{label} encoding must be one of {sorted(ROTATION_DIMS)}, got {value!r}"
        )
    return cast(RotationEncoding, value)


def opt_layout(value: object, label: str) -> ImageLayout:
    """Parse an optional image layout name, defaulting to ``hwc``."""
    if value is None:
        return "hwc"
    if value not in IMAGE_LAYOUTS:
        raise ValueError(f"{label} layout must be 'hwc' or 'chw', got {value!r}")
    return cast(ImageLayout, value)


__all__ = [
    "as_mapping",
    "load_json_mapping",
    "opt_encoding",
    "opt_layout",
    "opt_range",
    "require_mapping",
    "require_sequence",
    "require_str",
]
