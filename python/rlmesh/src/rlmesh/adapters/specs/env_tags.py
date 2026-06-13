"""The environment-side IO tags: sparse semantics over the env spaces.

An environment tags its observation entries and action layout with
*semantic roles* and the few facts the spaces cannot express (image axis
layout, rotation encoding, an explicit value range). Everything else --
keys, widths, dtypes, bounds -- is read from the gymnasium observation and
action spaces at resolve time by the native ``join`` step. This is the
asymmetry with the model side: models fully specify their payload
(:class:`~rlmesh.adapters.ModelSpec`), environments only tag.

The observation tags are keyed by their observation path (dotted
paths traverse nested ``Dict`` spaces), so an tag carries no key of
its own: the mapping key *is* the path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from ..constants import ENV_METADATA_KEY, IMAGE_PRIMARY, INSTRUCTION, JOINT_POS
from .action import ActionLayout
from .action_serialization import action_layout_from_dict, action_layout_to_dict
from .layouts import ImageLayout
from .rotations import RotationEncoding
from .serialization import (
    as_mapping,
    load_json_mapping,
    opt_encoding,
    opt_layout,
    opt_range,
    require_mapping,
    require_str,
)


@dataclass(frozen=True)
class ImageTag:
    """A camera image entry in an environment observation.

    Attributes:
        role: Semantic role used for matching, e.g. ``image/primary``.
        layout: Axis layout of the stored image.
        upside_down: Whether the image is rendered rotated 180 degrees
            relative to the canonical upright orientation.
    """

    role: str = IMAGE_PRIMARY
    layout: ImageLayout = "hwc"
    upside_down: bool = False


@dataclass(frozen=True)
class StateTag:
    """A numeric proprioception entry in an environment observation.

    Attributes:
        role: Semantic role used for matching, e.g. ``proprio/eef_pos``.
        encoding: Rotation encoding when the role is a rotation.
        range: Optional ``(low, high)`` value range; overrides any range
            the space's bounds would otherwise imply.
    """

    role: str = JOINT_POS
    encoding: RotationEncoding | None = None
    range: tuple[float, float] | None = None


@dataclass(frozen=True)
class TextTag:
    """A text entry (typically the task instruction) in an observation.

    Attributes:
        role: Semantic role used for matching.
    """

    role: str = INSTRUCTION


ObsTag: TypeAlias = ImageTag | StateTag | TextTag


def obs_tag_to_dict(tag: ObsTag) -> dict[str, Any]:
    """Return the JSON-compatible dict form of an observation tag."""
    if isinstance(tag, ImageTag):
        return {
            "type": "image",
            "role": tag.role,
            "layout": tag.layout,
            "upside_down": tag.upside_down,
        }
    if isinstance(tag, StateTag):
        return {
            "type": "state",
            "role": tag.role,
            "encoding": tag.encoding,
            "range": list(tag.range) if tag.range else None,
        }
    return {"type": "text", "role": tag.role}


def obs_tag_from_dict(item: object) -> ObsTag:
    """Build an observation tag from :func:`obs_tag_to_dict`."""
    data = as_mapping(item, "observation tag")
    kind = data.get("type")
    if kind == "image":
        return ImageTag(
            role=require_str(data, "role", "image tag"),
            layout=opt_layout(data.get("layout"), "image tag"),
            upside_down=bool(data.get("upside_down", False)),
        )
    if kind == "state":
        return StateTag(
            role=require_str(data, "role", "state tag"),
            encoding=opt_encoding(data.get("encoding"), "state tag"),
            range=opt_range(data.get("range"), "state tag"),
        )
    if kind == "text":
        return TextTag(role=require_str(data, "role", "text tag"))
    raise ValueError(f"unknown observation tag type {kind!r}")


@dataclass(frozen=True)
class EnvTags:
    """Declarative tags of an environment's observation and action.

    Attributes:
        observation: Observation tags keyed by observation path
            (dotted paths traverse nested ``Dict`` spaces).
        action: Layout of the action vector accepted by ``step``.
    """

    observation: Mapping[str, ObsTag]
    action: ActionLayout

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict form of these tags."""
        return {
            "observation": {
                key: obs_tag_to_dict(tag) for key, tag in self.observation.items()
            },
            "action": action_layout_to_dict(self.action),
        }

    def to_json(self) -> str:
        """Return these tags serialized as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata mapping fragment carrying these tags.

        Merge the result into env contract metadata so remote clients can
        recover the tags via :meth:`from_metadata`.
        """
        return {ENV_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvTags:
        """Build tags from :meth:`to_dict` output."""
        observation = {
            key: obs_tag_from_dict(value)
            for key, value in require_mapping(data, "observation").items()
        }
        return cls(
            observation=observation,
            action=action_layout_from_dict(require_mapping(data, "action")),
        )

    @classmethod
    def from_json(cls, payload: str) -> EnvTags:
        """Build tags from :meth:`to_json` output."""
        return cls.from_dict(load_json_mapping(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> EnvTags | None:
        """Extract tags from env contract metadata, if present."""
        payload = metadata.get(ENV_METADATA_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError(f"metadata key {ENV_METADATA_KEY!r} must hold a mapping")
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = [
    "EnvTags",
    "ImageTag",
    "ObsTag",
    "StateTag",
    "TextTag",
    "obs_tag_from_dict",
    "obs_tag_to_dict",
]
