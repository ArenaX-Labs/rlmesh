"""The environment-side IO tags: sparse semantics over the env spaces.

An environment tags its observation entries and action layout with
*semantic roles* and the few facts the spaces cannot express (image axis
layout, rotation encoding, an explicit value range). Everything else --
keys, widths, dtypes, bounds -- is read from the gymnasium observation and
action spaces at resolve time by the native ``join`` step. This is the
asymmetry with the model side: models fully specify their payload
(:class:`~rlmesh.adapters.ModelSpec`), environments only tag.

The observation tags are keyed by their observation path (dotted
paths traverse nested ``Dict`` spaces), so a tag carries no key of
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
    require_sequence,
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
        range: Optional ``(low, high)`` value range, supplying the bounds where
            the space leaves this leaf unbounded. If the space declares finite
            bounds that disagree with it, resolution errors rather than
            silently overriding them.
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


@dataclass(frozen=True)
class StateField:
    """One contiguous field of a flat numeric observation leaf.

    The observation-side mirror of
    :class:`~rlmesh.adapters.ActionComponent`: a slice of ``dim`` elements
    carrying a ``role``, with offsets implied by order within a
    :class:`StateLayout`. A field with no ``role`` is a *skip* -- it advances
    the offset but produces no feature, used to step over elements the model
    never consumes.

    Attributes:
        role: Semantic role matched against model state components, or None
            to skip this slice.
        dim: Number of elements this field occupies.
        encoding: Rotation encoding when the field is a rotation.
        range: Optional ``(low, high)`` value range for this field's slice,
            supplying the bounds where the space leaves it unbounded. If the
            space declares finite bounds for the slice that disagree with it,
            resolution errors rather than silently overriding them.
    """

    role: str | None = None
    dim: int = 0
    encoding: RotationEncoding | None = None
    range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"StateField.dim must be >= 1, got {self.dim}")
        if self.role is None and (self.encoding is not None or self.range is not None):
            raise ValueError(
                "StateField with no role is a skip and cannot carry an "
                "encoding or range"
            )


@dataclass(frozen=True, init=False)
class StateLayout:
    """An ordered split of one flat numeric observation leaf into role fields.

    The observation-side mirror of :class:`~rlmesh.adapters.ActionLayout`:
    fields are laid out in order, offsets accumulate, and the native ``join``
    requires the field widths to sum to the leaf width. Use it when an env
    returns a flat ``Box`` whose fixed index ranges carry distinct semantics
    (e.g. Metaworld)::

        StateLayout(StateField(EEF_POS, 3), StateField(GRIPPER, 1))

    Attributes:
        fields: State fields in vector order.
    """

    fields: tuple[StateField, ...]

    def __init__(self, *fields: StateField) -> None:
        if not fields:
            raise ValueError("StateLayout needs at least one StateField")
        roles = [field.role for field in fields if field.role is not None]
        if len(roles) != len(set(roles)):
            raise ValueError("StateLayout declares a role more than once")
        object.__setattr__(self, "fields", tuple(fields))


ObsTag: TypeAlias = ImageTag | StateTag | StateLayout | TextTag

# A bare tag/layout means "the observation is one leaf"; a mapping tags each
# key of a Dict observation. The bare form mirrors ``action`` being one layout.
ObsTags: TypeAlias = Mapping[str, ObsTag] | ObsTag


def _state_field_to_dict(field: StateField) -> dict[str, Any]:
    return {
        "role": field.role,
        "dim": field.dim,
        "encoding": field.encoding,
        "range": list(field.range) if field.range else None,
    }


def _state_field_from_dict(item: object) -> StateField:
    data = as_mapping(item, "state field")
    role = data.get("role")
    if role is not None and not isinstance(role, str):
        raise ValueError("state field role must be a string or null")
    raw_dim = data.get("dim", 0)
    # A missing or null dim flows through as 0 so StateField raises the clean
    # "dim must be >= 1" error rather than a raw TypeError from int(None).
    return StateField(
        role=role,
        dim=int(raw_dim) if raw_dim is not None else 0,
        encoding=opt_encoding(data.get("encoding"), "state field"),
        range=opt_range(data.get("range"), "state field"),
    )


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
    if isinstance(tag, StateLayout):
        return {
            "type": "layout",
            "fields": [_state_field_to_dict(field) for field in tag.fields],
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
    if kind == "layout":
        return StateLayout(
            *(
                _state_field_from_dict(field)
                for field in require_sequence(data, "fields")
            )
        )
    if kind == "text":
        return TextTag(role=require_str(data, "role", "text tag"))
    raise ValueError(f"unknown observation tag type {kind!r}")


@dataclass(frozen=True, init=False)
class EnvTags:
    """Declarative tags of an environment's observation and action.

    ``observation`` is either a mapping from observation path to its tag
    (dotted paths traverse nested ``Dict`` spaces), or -- when the observation
    is a single leaf -- a bare tag/layout, mirroring ``action`` being one
    layout. A bare tag is normalized to ``{".": tag}``, where ``"."`` is the
    reserved path for the flat/root observation, so the stored ``observation``
    is always a mapping.

    Attributes:
        observation: Observation tags keyed by observation path (always a
            mapping after construction; a bare tag is normalized to ``"."``).
        action: Layout of the action vector accepted by ``step``.
    """

    observation: Mapping[str, ObsTag]
    action: ActionLayout

    def __init__(self, observation: ObsTags, action: ActionLayout) -> None:
        normalized: Mapping[str, ObsTag] = (
            observation if isinstance(observation, Mapping) else {".": observation}
        )
        object.__setattr__(self, "observation", normalized)
        object.__setattr__(self, "action", action)

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
    "ObsTags",
    "StateField",
    "StateLayout",
    "StateTag",
    "TextTag",
    "obs_tag_from_dict",
    "obs_tag_to_dict",
]
