"""The environment-side tags: sparse semantics over the env spaces.

An environment tags its observation entries and action layout with
*semantic roles* and the few facts the spaces cannot express (image axis
layout, rotation encoding, an explicit value range). Everything else --
keys, widths, dtypes, bounds -- is read from the gymnasium observation and
action spaces at resolve time by the native ``join`` step. This is the
asymmetry with the model side: models fully specify their payload
(:class:`~rlmesh.adapters.ModelSpec`), environments only tag.

``observation`` is a recursive tree whose container type *is* the runtime
container type: a Python ``dict`` maps a ``Dict`` space, a Python ``tuple``
maps a ``Tuple`` space, and a bare leaf (an :class:`ImageTag`, :class:`StateTag`,
:class:`TextTag`, or :class:`Split`) tags a single space leaf. There are no
dotted keys and no magic root sentinel: nesting is real Python ``dict``
nesting, and a single-leaf observation is a bare leaf.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from ..constants import ENV_METADATA_KEY, IMAGE_PRIMARY, INSTRUCTION, JOINT_POS
from ._codec import normalize_spec, one_or_many, to_pair
from .action import Action
from .action_serialization import action_from_dict, action_to_dict
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
        encoding: Rotation encoding when the role is a rotation. A single
            encoding, or a sequence of them (the env's native first, then
            alternatives it can emit) for cross-version negotiation.
        range: Optional ``(low, high)`` value range, supplying the bounds where
            the space leaves this leaf unbounded. If the space declares finite
            bounds that disagree with it, resolution errors rather than
            silently overriding them.
    """

    role: str = JOINT_POS
    encoding: RotationEncoding | Sequence[RotationEncoding] | None = None
    range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoding", one_or_many(self.encoding))


@dataclass(frozen=True)
class TextTag:
    """A text entry (typically the task instruction) in an observation.

    Attributes:
        role: Semantic role used for matching.
    """

    role: str = INSTRUCTION


@dataclass(frozen=True)
class Field:
    """One contiguous field of a flat numeric observation leaf.

    The observation-side mirror of :class:`~rlmesh.adapters.Actuator`: a slice
    of ``dim`` elements carrying a ``role``, with offsets implied by order within
    a :class:`Split`. A field with no ``role`` is a *skip* -- it advances the
    offset but produces no feature, used to step over elements the model never
    consumes.

    Attributes:
        role: Semantic role matched against model state parts, or None to skip
            this slice.
        dim: Number of elements this field occupies.
        encoding: Rotation encoding when the field is a rotation. A single
            encoding, or a sequence of them (native first) for negotiation.
        range: Optional ``(low, high)`` value range for this field's slice,
            supplying the bounds where the space leaves it unbounded. If the
            space declares finite bounds for the slice that disagree with it,
            resolution errors rather than silently overriding them.
    """

    role: str | None = None
    dim: int = 0
    encoding: RotationEncoding | Sequence[RotationEncoding] | None = None
    range: tuple[float, float] | None = None
    # The `dim = 0` default only satisfies dataclass field ordering (the optional
    # `role` precedes it); 0 is never a valid width, so it is rejected at
    # construction below (matching the Rust Field codec's `dim >= 1` guard).
    # The role-less-skip rule (a skip carries no encoding/range) stays Rust-side.

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"Field {self.role!r}: dim must be >= 1, got {self.dim}")
        object.__setattr__(self, "encoding", one_or_many(self.encoding))


@dataclass(frozen=True, init=False)
class Split:
    """An ordered split of one flat numeric observation leaf into role fields.

    A *leaf*, not a container: one tensor split into role fields, the
    observation-side mirror of :class:`~rlmesh.adapters.Action`. Fields are laid
    out in order, offsets accumulate, and the native ``join`` requires the field
    widths to sum to the leaf width. Use it when an env returns a flat ``Box``
    whose fixed index ranges carry distinct semantics (e.g. Metaworld)::

        Split(Field(EEF_POS, 3), Field(GRIPPER, 1))

    Attributes:
        fields: State fields in vector order.
    """

    fields: tuple[Field, ...]

    def __init__(self, *fields: Field) -> None:
        if not fields:
            raise ValueError("Split needs at least one Field")
        roles = [field.role for field in fields if field.role is not None]
        if len(roles) != len(set(roles)):
            raise ValueError("Split declares a role more than once")
        object.__setattr__(self, "fields", tuple(fields))


# An observation leaf: tags a single space leaf.
ObsLeaf: TypeAlias = ImageTag | StateTag | TextTag | Split

# A recursive observation tree: a leaf, a Dict (mapping of str to subtree), or a
# Tuple (positional sequence of subtrees). The container type *is* the runtime
# container type. ``Mapping``/``Sequence`` here are the authored forms; a bare
# leaf is the single-leaf case.
ObsNode: TypeAlias = "ObsLeaf | Mapping[str, ObsNode] | tuple[ObsNode, ...]"

# Backwards-readable alias name retained for the env observation tree.
ObsTag: TypeAlias = ObsLeaf
ObsTags: TypeAlias = ObsNode


def _field_to_dict(field: Field) -> dict[str, Any]:
    return {
        "role": field.role,
        "dim": field.dim,
        "encoding": field.encoding,
        "range": list(field.range) if field.range else None,
    }


def _field_from_dict(item: Mapping[str, Any]) -> Field:
    # Canonical (Rust-validated) data: `dim` is present and >= 1.
    return Field(
        role=item.get("role"),
        dim=int(item["dim"]),
        encoding=one_or_many(item.get("encoding")),
        range=to_pair(item.get("range")),
    )


def _leaf_to_dict(tag: ObsLeaf) -> dict[str, Any]:
    """Return the JSON-compatible dict form of an observation leaf."""
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
    if isinstance(tag, Split):
        return {
            "type": "split",
            "fields": [_field_to_dict(field) for field in tag.fields],
        }
    return {"type": "text", "role": tag.role}


def obs_node_to_dict(node: ObsNode) -> Any:
    """Return the structural wire form of an observation tree node.

    A leaf becomes a dict carrying ``"type"``; a Python ``dict`` (a Dict node)
    becomes a plain object of recursively-encoded subnodes; a Python ``tuple``
    (a Tuple node) becomes a JSON array of recursively-encoded subnodes.
    """
    if isinstance(node, (ImageTag, StateTag, TextTag, Split)):
        return _leaf_to_dict(node)
    if isinstance(node, Mapping):
        return {key: obs_node_to_dict(child) for key, child in node.items()}
    if isinstance(node, tuple):
        return [obs_node_to_dict(child) for child in node]
    raise TypeError(
        f"observation node must be a leaf (ImageTag/StateTag/TextTag/Split), a "
        f"dict, or a tuple, got {type(node).__name__}"
    )


# The leaf-vocabulary `type` discriminants that mark a JSON object as a leaf
# rather than a Dict node (mirrors the Rust ``OBS_LEAF_TYPES``).
_OBS_LEAF_TYPES = frozenset({"image", "state", "text", "split"})


def _leaf_from_dict(item: Mapping[str, Any]) -> ObsLeaf:
    """Build an observation leaf from canonical (Rust-validated) dict form."""
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
            encoding=one_or_many(item.get("encoding")),
            range=to_pair(item.get("range")),
        )
    if kind == "split":
        return Split(*(_field_from_dict(field) for field in item["fields"]))
    if kind == "text":
        return TextTag(role=item["role"])
    raise ValueError(f"unknown observation leaf type {kind!r}")


def obs_node_from_dict(node: object) -> ObsNode:
    """Build an observation tree node from canonical (Rust-validated) form.

    Discrimination is structural: a list is a Tuple node, an object whose
    ``"type"`` is a leaf discriminant is a leaf, and any other object is a Dict
    node (the container type *is* the runtime container type).
    """
    if isinstance(node, list):
        return tuple(obs_node_from_dict(child) for child in node)
    if isinstance(node, Mapping):
        kind = node.get("type")
        if isinstance(kind, str) and kind in _OBS_LEAF_TYPES:
            return _leaf_from_dict(cast(Mapping[str, Any], node))
        return {key: obs_node_from_dict(child) for key, child in node.items()}
    raise TypeError(f"observation node must be an object or array, got {node!r}")


@dataclass(frozen=True)
class EnvTags:
    """Declarative tags of an environment's observation and action.

    ``observation`` is a recursive tree whose container type *is* the runtime
    container type: a bare leaf (the observation is one space leaf), a
    ``dict[str, subtree]`` (a ``Dict`` space), or a ``tuple`` of subtrees (a
    ``Tuple`` space). A leaf is an :class:`ImageTag`, :class:`StateTag`,
    :class:`TextTag`, or :class:`Split`.

    Attributes:
        observation: The observation tag tree.
        action: Layout of the action vector accepted by ``step``.
    """

    observation: ObsNode
    action: Action

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict form of these tags.

        The dataclass<->dict shape is built here, then validated and
        canonicalized by the authoritative Rust codec (the output is always the
        Rust-canonical form), so Python cannot emit a spec Rust would reject.
        """
        raw = {
            "observation": obs_node_to_dict(self.observation),
            "action": action_to_dict(self.action),
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
        return cls(
            observation=obs_node_from_dict(canonical["observation"]),
            action=action_from_dict(canonical["action"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> EnvTags:
        """Build tags from :meth:`to_json` output."""
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> EnvTags | None:
        """Extract tags from env contract metadata, or None when absent.

        Reads the single v1 metadata key (``rlmesh.adapters.v1.env_tags``). When
        a future v2 format lands it ships a new key and reader, restoring a
        newest-format-first dual read so a newer build still reads an older
        peer's v1 tags; that dispatch moves into the Rust codec (the single
        source of truth) once the PyO3 normalize door lands.
        """
        payload = metadata.get(ENV_METADATA_KEY)
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError(f"metadata key {ENV_METADATA_KEY!r} must hold a mapping")
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = [
    "EnvTags",
    "Field",
    "ImageTag",
    "ObsLeaf",
    "ObsNode",
    "ObsTag",
    "ObsTags",
    "Split",
    "StateTag",
    "TextTag",
]
