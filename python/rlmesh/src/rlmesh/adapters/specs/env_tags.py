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

from ..constants import ENV_METADATA_KEY
from ._codec import (
    check_accept_set,
    hashable_node,
    normalize_spec,
    one_or_many,
    to_pair,
)
from .action import Action
from .action_serialization import action_from_dict, action_to_dict
from .model_serialization import COMMON_LEAF_TYPES, decode_node, encode_node
from .vocabularies import ImageLayout, RotationEncoding


@dataclass(frozen=True)
class ImageTag:
    """A camera image entry in an environment observation.

    Attributes:
        role: Semantic role used for matching, e.g. ``image/primary``.
        layout: Axis layout of the stored image.
        upside_down: Whether the image is rendered rotated 180 degrees
            (a true rotation, not a vertical flip) relative to the canonical
            upright orientation. Declared on both ends; the adapter flips only
            when the env and the model disagree. (If a second orientation is
            ever needed, this should become a constrained string like ``dtype``,
            not a wider ``bool``.)
    """

    role: str
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

    role: str
    encoding: RotationEncoding | Sequence[RotationEncoding] | None = None
    range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoding", one_or_many(self.encoding))
        check_accept_set("StateTag", self.role, self.encoding)


@dataclass(frozen=True)
class TextTag:
    """A text entry (typically the task instruction) in an observation.

    Attributes:
        role: Semantic role used for matching (required, e.g. ``INSTRUCTION``).
    """

    role: str


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
        check_accept_set("Field", self.role, self.encoding)


@dataclass(frozen=True, init=False)
class Split:
    """An ordered split of one flat numeric observation leaf into role fields.

    A *leaf*, not a container: one tensor split into role fields, the
    observation-side mirror of :class:`~rlmesh.adapters.Action`. Fields are laid
    out in order, offsets accumulate, and the native ``join`` requires the field
    widths to sum to the leaf width. Use it when an env returns a flat ``Box``
    whose fixed index ranges carry distinct semantics (e.g. Metaworld)::

        Split(Field(EEF_POS, 3), Field(GRIPPER, 1))

    Construction rejects a role declared by more than one field. The
    authoritative Rust codec enforces the same rule at the wire door;
    duplicating it here is deliberate (the fail-fast-at-construction exception,
    like ``Field``'s ``dim >= 1`` check), so the author's own mistake surfaces
    at the authoring line instead of at serialize/resolve.

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


def _is_obs_leaf(node: object) -> bool:
    return isinstance(node, (ImageTag, StateTag, TextTag, Split))


def obs_node_to_dict(node: ObsNode) -> Any:
    """Return the structural wire form of an observation tree node.

    A leaf becomes a dict carrying ``"type"``; a Python ``dict`` (a Dict node)
    becomes a plain object of recursively-encoded subnodes; a Python ``tuple``
    (a Tuple node) becomes a JSON array of recursively-encoded subnodes.
    """
    return encode_node(node, _leaf_to_dict, _is_obs_leaf, "observation")


# The leaf-vocabulary `type` discriminants that mark a JSON object as a leaf
# rather than a Dict node (mirrors the Rust ``OBS_LEAF_TYPES``); the env side
# adds ``split`` to the shared common vocabulary.
_OBS_LEAF_TYPES = COMMON_LEAF_TYPES | {"split"}


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
    return cast(
        "ObsNode",
        decode_node(node, _leaf_from_dict, _OBS_LEAF_TYPES, "observation"),
    )


@dataclass(frozen=True)
class ObservationRoles:
    """The observation roles an environment declares, grouped by kind.

    Returned by :attr:`EnvTags.observation_roles` and
    :meth:`rlmesh.Session.observation_roles`. Each group lists role strings in
    declaration order; an env that declares no tags yields empty groups.

    Attributes:
        images: Roles of :class:`ImageTag` leaves.
        states: Roles of :class:`StateTag` leaves and of non-skip
            :class:`Field` slices inside a :class:`Split`.
        texts: Roles of :class:`TextTag` leaves.
    """

    images: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()


def _walk_observation_roles(
    node: object, images: list[str], states: list[str], texts: list[str]
) -> None:
    """Collect declared roles from an observation tag tree, in declaration order.

    Handles the three node shapes of ``ObsNode``: a Dict node (mapping), a Tuple
    node, and a bare leaf. A role-less (skip) :class:`Field` produces nothing.
    Any other node (e.g. a ``list`` container) raises the same ``TypeError`` the
    wire encoder raises for that tree, so the walker cannot silently accept a
    tree ``to_dict`` rejects.
    """
    if isinstance(node, Mapping):
        for child in cast("Mapping[str, Any]", node).values():
            _walk_observation_roles(child, images, states, texts)
    elif isinstance(node, tuple):
        for child in cast("tuple[Any, ...]", node):
            _walk_observation_roles(child, images, states, texts)
    elif isinstance(node, ImageTag):
        images.append(node.role)
    elif isinstance(node, StateTag):
        states.append(node.role)
    elif isinstance(node, Split):
        states.extend(f.role for f in node.fields if f.role is not None)
    elif isinstance(node, TextTag):
        texts.append(node.role)
    else:
        raise TypeError(
            f"observation node must be a leaf, a dict, or a tuple, got {node!r}"
        )


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

    def __post_init__(self) -> None:
        """Fail fast on a duplicate observation role at construction.

        Every consumer's resolve indexes env features by role per kind and
        rejects a duplicate, so tags that declare one can never resolve; catch
        it here (width-independently -- a vector env skips space-width checks
        but a duplicate role is invalid regardless), extending the same
        fail-fast-at-construction pattern as ``Split``'s duplicate-field check.
        """
        images: list[str] = []
        states: list[str] = []
        texts: list[str] = []
        _walk_observation_roles(self.observation, images, states, texts)
        for kind, roles in (("image", images), ("state", states), ("text", texts)):
            seen: set[str] = set()
            for role in roles:
                if role in seen:
                    raise ValueError(
                        f"env tags declare {kind} role {role!r} more than once"
                    )
                seen.add(role)

    def __hash__(self) -> int:
        """Hash consistently with the generated order-insensitive ``__eq__``.

        ``observation`` can be a Dict node (an unhashable Python ``dict``), so
        the dataclass-default field hash would fail even though the tags are
        frozen and compare by value -- and ``__eq__`` compares Dict nodes
        without regard to key order, so the hash must not depend on it either.
        Hash a key-order-canonical rendering (see :func:`hashable_node`).
        """
        return hash((hashable_node(self.observation), self.action))

    @property
    def observation_roles(self) -> ObservationRoles:
        """The declared observation roles, grouped by kind in declaration order.

        Walks the observation tag tree (dict/tuple containers and bare leaves):
        :class:`ImageTag` roles land in ``images``; :class:`StateTag` roles and
        non-skip :class:`Field` roles inside a :class:`Split` land in ``states``;
        :class:`TextTag` roles land in ``texts``.
        """
        images: list[str] = []
        states: list[str] = []
        texts: list[str] = []
        _walk_observation_roles(self.observation, images, states, texts)
        return ObservationRoles(
            images=tuple(images), states=tuple(states), texts=tuple(texts)
        )

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
            from ..resolver import AdapterResolutionError

            raise AdapterResolutionError(
                f"metadata key {ENV_METADATA_KEY!r} must hold a mapping"
            )
        return cls.from_dict(cast(Mapping[str, Any], payload))


__all__ = [
    "EnvTags",
    "Field",
    "ImageTag",
    "ObsLeaf",
    "ObsNode",
    "ObservationRoles",
    "Split",
    "StateTag",
    "TextTag",
]
