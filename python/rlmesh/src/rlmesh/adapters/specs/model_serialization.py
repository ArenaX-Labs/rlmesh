"""Structural tree round-trip for model input nodes.

The dataclass<->dict *shape* lives here; validation and canonicalization are done
by the authoritative Rust codec (see :mod:`._codec`), so the from-dict reader
operates on already-valid canonical data.

Discrimination on the wire is structural (mirroring the Rust ``InputNode``): a
JSON array is a Tuple, a JSON object whose ``"type"`` is a leaf discriminant
(``image``/``state``/``text``/``custom``) is a leaf, and any other JSON object is
a Dict. Placement in the tree is the payload position -- there is no ``key``.

The generic structural tree-walkers (:func:`encode_node` / :func:`decode_node`)
and the shared leaf-vocabulary base (:data:`COMMON_LEAF_TYPES`) live here and are
reused by the observation side (:mod:`.env_tags`), which differs only in its leaf
encoder/decoder and its one extra leaf discriminant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from ._codec import one_or_many, to_pair
from .custom_encoding import CustomEncoding
from .model_inputs import (
    Concat,
    ConcatPart,
    Custom,
    Image,
    InputNode,
    ModelLeaf,
    State,
    Text,
)

# The leaf-vocabulary `type` discriminants shared by both sides; each side adds
# its one own leaf (model: `custom`, env: `split`). Single-sourced here so the
# two sides cannot drift on the common vocabulary. Package-internal (imported by
# :mod:`.env_tags`) but not part of the public API, hence no leading underscore.
COMMON_LEAF_TYPES = frozenset({"image", "state", "text"})

# The leaf-vocabulary `type` discriminants that mark a JSON object as a leaf
# rather than a Dict node (mirrors the Rust ``MODEL_LEAF_TYPES``).
_MODEL_LEAF_TYPES = COMMON_LEAF_TYPES | {"custom"}


def encode_node(
    node: object,
    encode_leaf: Callable[[Any], dict[str, Any]],
    is_leaf: Callable[[object], bool],
    what: str,
) -> Any:
    """Encode a recursive spec tree node to its structural wire form.

    A leaf (``is_leaf(node)`` true) becomes a dict carrying ``"type"`` via
    ``encode_leaf``; a Python ``Mapping`` (a Dict node) becomes a plain object of
    recursively-encoded subnodes; a Python ``tuple`` (a Tuple node) becomes a JSON
    array. ``what`` names the side for the error message.
    """
    if is_leaf(node):
        return encode_leaf(node)
    if isinstance(node, Mapping):
        mapping = cast("Mapping[str, Any]", node)
        return {
            key: encode_node(child, encode_leaf, is_leaf, what)
            for key, child in mapping.items()
        }
    if isinstance(node, tuple):
        children = cast("tuple[Any, ...]", node)
        return [encode_node(child, encode_leaf, is_leaf, what) for child in children]
    raise TypeError(f"{what} node must be a leaf, a dict, or a tuple, got {node!r}")


def decode_node(
    node: object,
    decode_leaf: Callable[[Mapping[str, Any]], Any],
    leaf_types: frozenset[str],
    what: str,
) -> Any:
    """Decode a structural wire form back to a recursive spec tree node.

    Discrimination is structural: a list is a Tuple node, an object whose
    ``"type"`` is in ``leaf_types`` is a leaf (decoded via ``decode_leaf``), and
    any other object is a Dict node (the container type *is* the runtime
    container type). ``what`` names the side for the error message.
    """
    if isinstance(node, list):
        items = cast("list[Any]", node)
        return tuple(
            decode_node(child, decode_leaf, leaf_types, what) for child in items
        )
    if isinstance(node, Mapping):
        mapping = cast("Mapping[str, Any]", node)
        kind = mapping.get("type")
        if isinstance(kind, str) and kind in leaf_types:
            return decode_leaf(mapping)
        return {
            key: decode_node(child, decode_leaf, leaf_types, what)
            for key, child in mapping.items()
        }
    raise TypeError(f"{what} node must be an object or array, got {node!r}")


def _image_to_dict(item: Image) -> dict[str, Any]:
    image: dict[str, Any] = {
        "type": "image",
        "role": item.role,
        "height": item.height,
        "width": item.width,
        "layout": item.layout,
        "dtype": item.dtype,
        "normalize": item.normalize,
        "lead_dims": item.lead_dims,
        "upside_down": item.upside_down,
        "resample": item.resample,
    }
    # Additive over the pinned v1 wire (the env still sends one frame/step),
    # emitted only when set. It IS consumed by the native adapter engine
    # (episode-keyed FrameBuffers), not host-only -- see Image.stack.
    if item.stack != 1:
        image["stack"] = item.stack
    # Opt-in, emitted only when set (byte-parity with the pinned wire format).
    if item.allow_upscale:
        image["allow_upscale"] = True
    if item.fit is not None:
        image["fit"] = item.fit
    if item.channels is not None:
        image["channels"] = item.channels
    if item.normalize_range is not None:
        image["normalize_range"] = list(item.normalize_range)
    if item.optional:
        image["optional"] = True
    if item.absent_fill is not None:
        image["absent_fill"] = item.absent_fill
    return image


def _part_to_dict(part: State) -> Any:
    """Return the wire form of one concat part: a bare role string or object."""
    if isinstance(part.encoding, CustomEncoding):
        raise ValueError(
            f"state part {part.role!r} uses a CustomEncoding, whose host-side "
            "transforms cannot be serialized; resolve it locally (the model spec "
            "need not travel), or add the encoding to the native vocabulary for "
            "a shared convention"
        )
    role_only = (
        part.encoding is None
        and part.dim is None
        and part.index is None
        and part.range is None
        and not part.optional
    )
    if role_only:
        return part.role
    out: dict[str, Any] = {"role": part.role}
    if part.encoding is not None:
        out["encoding"] = part.encoding
    if part.dim is not None:
        out["dim"] = part.dim
    if part.index is not None:
        out["index"] = part.index
    if part.optional:
        out["optional"] = True
    if part.range is not None:
        out["range"] = list(part.range)
    return out


def _state_parts(item: State | Concat) -> tuple[State, ...]:
    """The concat parts of a State (one) or Concat (several), as ``State``s.

    A bare ``State`` leaf is its own single part; its container fields belong to
    the leaf and are emitted at the leaf level by :func:`_state_to_dict`. A
    ``Concat`` part may only carry part fields: a part's container fields cannot
    be represented in the Rust ``ConcatPart`` and would be silently dropped, so a
    non-default one is rejected here (an honest authoring-time error).
    """
    if isinstance(item, State):
        return (item,)
    parts: list[State] = []
    for part in item.parts:
        if isinstance(part, State):
            _check_part_container_default(part)
        parts.append(State(part) if isinstance(part, str) else part)
    return tuple(parts)


def _check_part_container_default(part: State) -> None:
    """Reject a Concat part that carries non-default container fields.

    A part contributes only part fields (role/encoding/dim/index/optional/range);
    its assembly options (``pad_to``/``dtype``/``reshape``/``container``) live on
    the enclosing :class:`Concat`, not the part, and the Rust ``ConcatPart`` has
    no slot for them. Serializing would silently drop them, so raise instead.
    """
    offenders = [
        name
        for name, value, default in (
            ("pad_to", part.pad_to, None),
            ("dtype", part.dtype, "float32"),
            ("reshape", part.reshape, None),
            ("container", part.container, "array"),
        )
        if value != default
    ]
    if offenders:
        raise ValueError(
            f"State part {part.role!r} sets {', '.join(offenders)}, but a Concat "
            "part carries only part fields (role/encoding/dim/index/optional/"
            "range); set the assembly options (pad_to/dtype/reshape/container) on "
            "the enclosing Concat instead"
        )


def _state_to_dict(item: State | Concat) -> dict[str, Any]:
    return {
        "type": "state",
        "components": [_part_to_dict(part) for part in _state_parts(item)],
        "pad_to": item.pad_to,
        "dtype": item.dtype,
        # `()` is a valid 0-D scalar target; only None means "no reshape".
        "reshape": list(item.reshape) if item.reshape is not None else None,
        "container": item.container,
    }


def model_leaf_to_dict(item: ModelLeaf) -> dict[str, Any]:
    """Return the JSON-compatible dict form of a model input leaf."""
    if isinstance(item, Image):
        return _image_to_dict(item)
    if isinstance(item, (State, Concat)):
        return _state_to_dict(item)
    if isinstance(item, Text):
        return {
            "type": "text",
            "role": item.role,
            "container": item.container,
            "default": item.default,
        }
    if isinstance(item, Custom):
        if item.entrypoint is None:
            # Publish boundary: an in-process callable cannot be serialized.
            raise ValueError(
                "custom input holds an in-process callable and cannot be "
                "serialized; use Custom(entrypoint='module:callable') instead"
            )
        # An entrypoint custom carries an importable `module:callable`, so
        # publishing it would ship executable code in a contract a remote
        # consumer might import. Until an allowlist-based trust model exists,
        # refuse to serialize it here. Local resolve still runs it (gated by
        # trust_entrypoints) via the resolver's _model_wire, which handles
        # customs itself and never routes them through this serializer -- so this
        # gate is publish-only.
        raise ValueError(
            f"custom input carries an entrypoint ({item.entrypoint!r}) and "
            "cannot be published in v1 contract metadata; resolve the spec "
            "locally (the model spec need not travel), or wait for the "
            "allowlist-gated trust system"
        )
    raise TypeError(f"unknown model input leaf {type(item).__name__}")


def _is_model_leaf(node: object) -> bool:
    return isinstance(node, (Image, State, Concat, Text, Custom))


def model_input_to_dict(node: InputNode) -> Any:
    """Return the structural wire form of a model input tree node.

    A leaf becomes a dict carrying ``"type"``; a Python ``dict`` (a Dict node)
    becomes a plain object of recursively-encoded subnodes; a Python ``tuple``
    (a Tuple node) becomes a JSON array.
    """
    return encode_node(node, model_leaf_to_dict, _is_model_leaf, "model input")


def _part_from_dict(item: object) -> ConcatPart:
    """Build one concat part from canonical form: a bare role or a State."""
    if isinstance(item, str):
        return item
    part = cast(Mapping[str, Any], item)
    return State(
        role=part["role"],
        encoding=one_or_many(part.get("encoding")),
        dim=part.get("dim"),
        index=part.get("index"),
        optional=bool(part.get("optional", False)),
        range=to_pair(part.get("range")),
    )


def model_leaf_from_dict(data: Mapping[str, Any]) -> ModelLeaf:
    """Build a model input leaf from canonical (Rust-validated) dict form."""
    kind = data["type"]
    if kind == "image":
        return Image(
            role=data["role"],
            height=data.get("height"),
            width=data.get("width"),
            layout=data.get("layout", "hwc"),
            channels=data.get("channels"),
            dtype=data.get("dtype", "uint8"),
            normalize=bool(data.get("normalize", False)),
            normalize_range=to_pair(data.get("normalize_range")),
            optional=bool(data.get("optional", False)),
            absent_fill=data.get("absent_fill"),
            lead_dims=int(data.get("lead_dims", 0)),
            upside_down=bool(data.get("upside_down", False)),
            resample=data.get("resample", "bilinear_aa"),
            allow_upscale=bool(data.get("allow_upscale", False)),
            fit=one_or_many(data.get("fit")),
            stack=int(data.get("stack", 1)),
        )
    if kind == "state":
        reshape = data.get("reshape")
        reshape_t = None if reshape is None else tuple(int(n) for n in reshape)
        pad_to = data.get("pad_to")
        dtype = data.get("dtype", "float32")
        container = data.get("container", "array")
        # `_part_from_dict` yields a bare role ``str`` for a role-only wire
        # component and a parameterized ``State`` for an object component.
        parts = tuple(_part_from_dict(part) for part in data["components"])
        # A single *parameterized* part round-trips to a bare State carrying the
        # leaf's container fields: a State and a one-part Concat over the same
        # parameterized part serialize identically, and State is the natural
        # author for the single-part case (the documented wire equivalence). A
        # single *bare* role string, or more than one part, is a genuine Concat --
        # collapsing those would change the authored leaf type (a bare role-only
        # Concat, and a multi-part Concat, must survive as Concat).
        if len(parts) == 1 and isinstance(parts[0], State):
            base = parts[0]
            return State(
                role=base.role,
                encoding=base.encoding,
                dim=base.dim,
                index=base.index,
                optional=base.optional,
                range=base.range,
                pad_to=pad_to,
                dtype=dtype,
                reshape=reshape_t,
                container=container,
            )
        return Concat(
            *parts,
            pad_to=pad_to,
            dtype=dtype,
            reshape=reshape_t,
            container=container,
        )
    if kind == "text":
        return Text(
            role=data["role"],
            container=data.get("container", "str"),
            default=data.get("default"),
        )
    if kind == "custom":
        # The resolver's internal `transform: "host:<key>"` placeholder (see
        # resolver._model_wire) references a host-side callable that stayed local
        # during resolve; it is NOT an importable entrypoint and is never meant
        # to round-trip back through from_dict. Reject it rather than minting a
        # Custom with a bogus `host:` entrypoint a later trust_entrypoints
        # resolve would try to import.
        transform = data["transform"]
        if isinstance(transform, str) and transform.startswith("host:"):
            raise ValueError(
                f"custom input carries a host-placeholder transform "
                f"({transform!r}); it references a host-side callable that "
                "stayed local during resolve and cannot be reconstructed from "
                "the wire form -- resolve the spec locally instead of routing "
                "its internal wire form back through from_dict"
            )
        return Custom(entrypoint=transform)
    raise ValueError(f"unknown model input type {kind!r}")


def model_input_from_dict(node: object) -> InputNode:
    """Build a model input tree node from canonical (Rust-validated) form.

    Discrimination is structural: a list is a Tuple, an object whose ``"type"``
    is a leaf discriminant is a leaf, any other object is a Dict.
    """
    return cast(
        "InputNode",
        decode_node(node, model_leaf_from_dict, _MODEL_LEAF_TYPES, "model input"),
    )


__all__ = [
    "decode_node",
    "encode_node",
    "model_input_from_dict",
    "model_input_to_dict",
    "model_leaf_from_dict",
    "model_leaf_to_dict",
]
