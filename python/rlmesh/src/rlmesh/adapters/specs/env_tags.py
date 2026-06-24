"""The environment-side tags: sparse semantics over the env spaces.

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
from ._codec import normalize_spec, to_pair
from .action import ActionLayout
from .action_serialization import action_layout_from_dict, action_layout_to_dict
from .vocabularies import ImageLayout, RotationEncoding


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
    # dim >= 1 and the role-less-skip rule (a skip carries no encoding/range) are
    # enforced by the Rust codec (StateField's TryFrom guard) at serialize/normalize.


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


def _state_field_from_dict(item: Mapping[str, Any]) -> StateField:
    # Canonical (Rust-validated) data: `dim` is present and >= 1.
    return StateField(
        role=item.get("role"),
        dim=int(item["dim"]),
        encoding=item.get("encoding"),
        range=to_pair(item.get("range")),
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


def obs_tag_from_dict(item: Mapping[str, Any]) -> ObsTag:
    """Build an observation tag from canonical (Rust-validated) dict form."""
    kind = item["type"]
    if kind == "image":
        return ImageTag(
            role=item["role"],
            layout=item.get("layout", "hwc"),
            upside_down=bool(item.get("upside_down", False)),
        )
    if kind == "state":
        return StateTag(
            role=item["role"],
            encoding=item.get("encoding"),
            range=to_pair(item.get("range")),
        )
    if kind == "layout":
        return StateLayout(*(_state_field_from_dict(field) for field in item["fields"]))
    if kind == "text":
        return TextTag(role=item["role"])
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
        """Return a JSON-compatible dict form of these tags.

        The dataclass<->dict shape is built here, then validated and
        canonicalized by the authoritative Rust codec (the output is always the
        Rust-canonical form), so Python cannot emit a spec Rust would reject.
        """
        raw = {
            "observation": {
                key: obs_tag_to_dict(tag) for key, tag in self.observation.items()
            },
            "action": action_layout_to_dict(self.action),
        }
        return normalize_spec("env", raw, allow_custom=True)

    def to_json(self) -> str:
        """Return these tags serialized as a JSON string."""
        # allow_nan=False: refuse to emit the non-RFC-8259 `Infinity`/`NaN`
        # tokens the Rust serde codec rejects (a directly-constructed dataclass
        # bypasses the from_dict finiteness guards).
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata mapping fragment carrying these tags.

        Merge the result into env contract metadata so remote clients can
        recover the tags via :meth:`from_metadata`.
        """
        return {ENV_METADATA_KEY: self.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnvTags:
        """Build tags from :meth:`to_dict` output.

        The input is validated and canonicalized by the Rust codec first, so the
        Python shape readers below operate on already-valid data.
        """
        canonical = normalize_spec("env", data, allow_custom=True)
        observation = {
            key: obs_tag_from_dict(value)
            for key, value in canonical["observation"].items()
        }
        return cls(
            observation=observation,
            action=action_layout_from_dict(canonical["action"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> EnvTags:
        """Build tags from :meth:`to_json` output."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> EnvTags | None:
        """Extract tags from env contract metadata, newest format first.

        Iterates the known metadata keys newest-format-first: a future v2 format
        ships a new key (``rlmesh.adapters.v2.env_tags`` -> a v2 reader) prepended
        to this list, so a newer build still reads an older peer's v1 tags. This
        is the single dual-read dispatch the v1->v2 rule promises; it moves into
        the Rust codec (the single source of truth) once the PyO3 normalize door
        lands.
        """
        readers = ((ENV_METADATA_KEY, cls.from_dict),)
        for key, reader in readers:
            payload = metadata.get(key)
            if payload is None:
                continue
            if not isinstance(payload, Mapping):
                raise TypeError(f"metadata key {key!r} must hold a mapping")
            return reader(cast(Mapping[str, Any], payload))
        return None


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
