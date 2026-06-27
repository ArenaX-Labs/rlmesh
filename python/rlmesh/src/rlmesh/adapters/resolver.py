"""Resolve env tags and a model spec into a concrete adapter.

Resolution and plan application run in the native ``rlmesh-adapters`` core
(the same implementation behind every language binding); this module
serializes the tags and spec across the boundary, hands the
gymnasium observation/action spaces to the native projector, and keeps the
host-language concerns where they belong: entrypoint trust gating, custom
callables, and the error type.

Custom-input host holes and custom-encoding shims are keyed by the leaf's
*structured placement path* in the input tree -- a tuple of ``str`` (Dict key)
and ``int`` (Tuple index) segments, mirroring the native ``NodePath`` (the empty
tuple is the root of a bare-leaf input). The rendered ``host:<placement>`` string
goes into the wire spec only; nothing parses it back.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from .._entrypoint import resolve_entrypoint
from .._rlmesh import ROTATION_DIMS, adapters_resolve
from .adapter import ActEncShim, Adapter, ObsEncShim
from .constants import ENV_METADATA_KEY
from .specs import (
    Action,
    Actuator,
    Concat,
    Custom,
    CustomEncoding,
    EnvTags,
    Image,
    InputNode,
    ModelSpec,
    ObsTransform,
    RotationTransform,
    State,
    Text,
)
from .specs.action_serialization import action_to_dict
from .specs.model_serialization import model_input_to_dict

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..specs import EnvContract


class AdapterResolutionError(ValueError):
    """Raised when env tags and a model spec cannot be reconciled."""


# Representative valid value per native base encoding, used by the optional
# inverse self-check to confirm a CustomEncoding's two arms round-trip.
_PROBES: dict[str, list[float]] = {
    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    "axis_angle": [0.1, -0.2, 0.3],
    "euler_xyz": [0.1, -0.2, 0.3],
    "rot6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "rot6d_rowmajor": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
}


def _placement(segments: tuple[str | int, ...]) -> str:
    """Render a tree position as the canonical native ``NodePath`` string.

    Mirrors ``rlmesh_adapters::path::NodePath`` ``Display``: dot-joined keys,
    ``[i]`` for tuple indices, and ``<root>`` for the empty path (a bare leaf).
    """
    if not segments:
        return "<root>"
    out = ""
    for position, segment in enumerate(segments):
        if isinstance(segment, int):
            out += f"[{segment}]"
        else:
            out += ("." if position > 0 else "") + segment
    return out


def _is_leaf(node: object) -> bool:
    return isinstance(node, (Image, State, Concat, Text, Custom))


def _walk_tree(
    node: InputNode,
    *,
    on_leaf: Callable[[tuple[str | int, ...], object], Any],
    on_dict: Callable[[dict[str, Any]], Any],
    on_tuple: Callable[[list[Any]], Any],
    segments: tuple[str | int, ...] = (),
) -> Any:
    """Recurse the input tree, calling the per-node/per-leaf callbacks.

    The single tree-walk every resolver traversal is built on. The segment walk
    mirrors the native ``collect_leaves`` (a ``Dict`` pushes the key, a ``Tuple``
    pushes the index), so a rendered placement matches the core's. Leaf detection
    is by ``isinstance`` against the public leaf dataclasses, never a string
    vocabulary. The three node callbacks decide the result shape:

    - ``on_leaf(segments, leaf)`` -- the value produced for each leaf;
    - ``on_dict(children)`` -- combines a ``{key: result}`` map;
    - ``on_tuple(children)`` -- combines an ordered ``[result, ...]`` list.
    """

    def recurse(node: InputNode, segments: tuple[str | int, ...]) -> Any:
        if _is_leaf(node):
            return on_leaf(segments, node)
        if isinstance(node, Mapping):
            return on_dict(
                {key: recurse(child, (*segments, key)) for key, child in node.items()}
            )
        if isinstance(node, tuple):
            return on_tuple(
                [recurse(child, (*segments, index)) for index, child in enumerate(node)]
            )
        raise TypeError(f"unexpected input tree node {type(node).__name__}")

    return recurse(node, segments)


def _iter_leaves(
    node: InputNode, segments: tuple[str | int, ...] = ()
) -> Iterator[tuple[tuple[str | int, ...], object]]:
    """Yield ``(segments, leaf)`` for each leaf in the input tree, DFS order.

    ``segments`` seeds the walk's path prefix (default: the root).
    """
    leaves: list[tuple[tuple[str | int, ...], object]] = []
    _walk_tree(
        node,
        on_leaf=lambda path, leaf: leaves.append((path, leaf)),
        on_dict=lambda _children: None,
        on_tuple=lambda _children: None,
        segments=segments,
    )
    return iter(leaves)


def _map_leaves(node: InputNode, transform: Any) -> InputNode:
    """Rebuild the input tree, replacing each leaf with ``transform(seg, leaf)``.

    Containers keep their declared kind (``Dict`` -> dict, ``Tuple`` -> tuple).
    """
    return cast(
        InputNode,
        _walk_tree(
            node,
            on_leaf=transform,
            on_dict=lambda children: children,
            on_tuple=lambda children: tuple(children),
        ),
    )


def _check_native_encoding(encoding: object, where: str) -> None:
    """Reject a non-native encoding string.

    Closes the gap where a misspelled native name constructed directly in
    Python bypasses the ``Literal`` check.
    """
    if isinstance(encoding, str) and encoding not in ROTATION_DIMS:
        raise AdapterResolutionError(
            f"{where} has unknown rotation encoding {encoding!r}; expected one "
            f"of {sorted(ROTATION_DIMS)} or a CustomEncoding"
        )


def _shadow_state(
    segments: tuple[str | int, ...], state: State, obs_shims: list[ObsEncShim]
) -> State:
    """Replace a custom-encoded single-part state with its base-encoding shadow.

    Records an observation shim keyed by the structured placement path. A custom
    encoding must be the sole content of a single-part :class:`State`, with no
    width-altering or assembly options.
    """
    placement = _placement(segments)
    _check_native_encoding(state.encoding, f"state input {placement!r}")
    if not isinstance(state.encoding, CustomEncoding):
        return state
    encoding = state.encoding
    if state.dim is not None or state.index is not None:
        raise AdapterResolutionError(
            f"state input {placement!r}: a CustomEncoding part cannot also set "
            "dim or index (they would change its width)"
        )
    if state.optional:
        raise AdapterResolutionError(
            f"state input {placement!r}: a CustomEncoding part cannot be optional"
        )
    if state.pad_to is not None or state.reshape is not None:
        raise AdapterResolutionError(
            f"state input {placement!r}: pad_to/reshape run before the host-side "
            "encoding shim and would break it; drop them"
        )
    if state.container != "array":
        raise AdapterResolutionError(
            f"state input {placement!r}: a CustomEncoding requires container='array'"
        )
    if encoding.is_entrypoint:
        raise AdapterResolutionError(
            f"state input {placement!r}: entrypoint CustomEncoding is not yet "
            "supported; use in-process callables for now"
        )
    if encoding.from_base is None:
        raise AdapterResolutionError(
            f"state input {placement!r}: an observation CustomEncoding needs from_base"
        )
    obs_shims.append(
        ObsEncShim(
            segments=segments,
            base=encoding.base,
            width=encoding.width,
            name=encoding.name,
            dtype=state.dtype,
            from_base=cast("RotationTransform", encoding.from_base),
        )
    )
    return replace(state, encoding=encoding.base)


def _reject_concat_custom_encoding(
    segments: tuple[str | int, ...], concat: Concat
) -> None:
    """A CustomEncoding inside a multi-part Concat is unsupported (env offsets)."""
    placement = _placement(segments)
    for part in concat.parts:
        encoding = part.encoding if isinstance(part, State) else None
        _check_native_encoding(encoding, f"state input {placement!r}")
        if isinstance(encoding, CustomEncoding):
            raise AdapterResolutionError(
                f"state input {placement!r} uses a CustomEncoding, which must be "
                "the sole part of a single-part State (observation offsets are "
                "env-dependent); give the rotation its own input slot"
            )


def _shadow_action(action: Action, act_shims: list[ActEncShim]) -> Action:
    """Replace custom-encoded action actuators with base-encoding shadows.

    Records action shims with model-declared offsets.
    """
    shadow_components: list[Actuator] = []
    offset = 0
    for component in action.components:
        _check_native_encoding(component.encoding, f"actuator {component.role!r}")
        if isinstance(component.encoding, CustomEncoding):
            encoding = component.encoding
            if component.binary:
                raise AdapterResolutionError(
                    f"actuator {component.role!r}: a CustomEncoding cannot be binary"
                )
            if encoding.is_entrypoint:
                raise AdapterResolutionError(
                    f"actuator {component.role!r}: entrypoint CustomEncoding is "
                    "not yet supported; use in-process callables for now"
                )
            if encoding.to_base is None:
                raise AdapterResolutionError(
                    f"actuator {component.role!r}: an action CustomEncoding needs to_base"
                )
            act_shims.append(
                ActEncShim(
                    offset=offset,
                    base=encoding.base,
                    width=component.dim,
                    name=encoding.name,
                    to_base=cast("RotationTransform", encoding.to_base),
                )
            )
            shadow_components.append(replace(component, encoding=encoding.base))
        else:
            shadow_components.append(component)
        offset += component.dim
    return Action(*shadow_components, clip=action.clip)


def _custom_encodings(model_spec: ModelSpec) -> Iterator[CustomEncoding]:
    for _segments, leaf in _iter_leaves(model_spec.input):
        if isinstance(leaf, State) and isinstance(leaf.encoding, CustomEncoding):
            yield leaf.encoding
        elif isinstance(leaf, Concat):
            for part in leaf.parts:
                if isinstance(part, State) and isinstance(
                    part.encoding, CustomEncoding
                ):
                    yield part.encoding
    for component in model_spec.output.components:
        if isinstance(component.encoding, CustomEncoding):
            yield component.encoding


def _check_inverses(model_spec: ModelSpec) -> None:
    """Round-trip each two-armed inline CustomEncoding on a probe.

    Catches a mispaired encode/decode (silent train/serve skew).
    """
    import numpy as np

    seen: set[int] = set()
    for encoding in _custom_encodings(model_spec):
        if id(encoding) in seen:
            continue
        seen.add(id(encoding))
        if (
            encoding.is_entrypoint
            or encoding.from_base is None
            or encoding.to_base is None
        ):
            continue
        probe = _PROBES.get(encoding.base)
        if probe is None:
            continue
        from_base = cast("RotationTransform", encoding.from_base)
        to_base = cast("RotationTransform", encoding.to_base)
        base = np.asarray(probe, dtype=np.float64)
        try:
            custom = np.asarray(from_base(base), dtype=np.float64)
            roundtrip = np.asarray(to_base(custom), dtype=np.float64)
        except Exception as exc:
            raise AdapterResolutionError(
                f"CustomEncoding {encoding.name!r} raised during the inverse "
                f"self-check: {exc}"
            ) from exc
        if roundtrip.shape != base.shape or not np.allclose(roundtrip, base, atol=1e-5):
            raise AdapterResolutionError(
                f"CustomEncoding {encoding.name!r} arms are not inverses: "
                f"to_base(from_base({probe})) = {roundtrip.tolist()}, expected "
                f"{probe}; pass resolve(..., check_inverse=False) to skip"
            )


def _substitute_encodings(
    model_spec: ModelSpec,
) -> tuple[ModelSpec, tuple[ObsEncShim, ...], tuple[ActEncShim, ...]]:
    """Return a base-substituted shadow spec plus host-side encoding shims.

    The shadow spec lets the native core see only known encodings; the shims
    repack the custom fields at the boundary, keyed by placement path.
    """
    obs_shims: list[ObsEncShim] = []
    act_shims: list[ActEncShim] = []

    def shadow_leaf(segments: tuple[str | int, ...], leaf: object) -> object:
        if isinstance(leaf, State):
            return _shadow_state(segments, leaf, obs_shims)
        if isinstance(leaf, Concat):
            _reject_concat_custom_encoding(segments, leaf)
            return leaf
        return leaf

    shadow_input = _map_leaves(model_spec.input, shadow_leaf)
    shadow_action = _shadow_action(model_spec.output, act_shims)
    shadow = ModelSpec(input=shadow_input, output=shadow_action)
    return shadow, tuple(obs_shims), tuple(act_shims)


def _model_wire(
    model_spec: ModelSpec, *, trust_entrypoints: bool
) -> tuple[dict[str, Any], dict[tuple[str | int, ...], ObsTransform]]:
    """Split a model spec into its wire form and the local custom callables.

    Custom inputs never cross to the native core as code: an entrypoint
    string is gated on ``trust_entrypoints`` and imported here; an
    in-process callable stays here and is referenced by a ``host:<placement>``
    placeholder. Either way the native plan keeps the input as a hole that
    :class:`Adapter` fills from the raw observation. The returned callables are
    keyed by the structured placement path (a ``str``/``int`` segment tuple); the
    rendered ``host:<placement>`` string lives only in the wire spec, never
    parsed back.
    """
    customs: dict[tuple[str | int, ...], ObsTransform] = {}

    def wire_leaf(segments: tuple[str | int, ...], leaf: object) -> Any:
        if isinstance(leaf, Custom):
            if leaf.transform is not None:
                customs[segments] = leaf.transform
                wire_transform = f"host:{_placement(segments)}"
            else:
                entrypoint = cast(str, leaf.entrypoint)
                if not trust_entrypoints:
                    raise AdapterResolutionError(
                        f"custom input at {_placement(segments)!r} references "
                        f"entrypoint {entrypoint!r}; pass "
                        "resolve(..., trust_entrypoints=True) to allow importing it"
                    )
                customs[segments] = cast(
                    ObsTransform,
                    resolve_entrypoint(entrypoint, label="custom input transform"),
                )
                wire_transform = entrypoint
            return {"type": "custom", "transform": wire_transform}
        return model_input_to_dict(cast(InputNode, leaf))

    # The structural wire form: a Tuple serializes to a JSON list, a Dict to a
    # JSON object, each leaf via ``wire_leaf``.
    wire_input = _walk_tree(
        model_spec.input,
        on_leaf=wire_leaf,
        on_dict=lambda children: children,
        on_tuple=lambda children: children,
    )
    wire = {"input": wire_input, "output": action_to_dict(model_spec.output)}
    return wire, customs


def _image_stacks(model_spec: ModelSpec) -> dict[tuple[str | int, ...], int]:
    """Frame-stack depths the model wants, keyed by structured placement (>1)."""
    stacks: dict[tuple[str | int, ...], int] = {}
    for segments, leaf in _iter_leaves(model_spec.input):
        if isinstance(leaf, Image) and leaf.stack > 1:
            stacks[segments] = leaf.stack
    return stacks


def resolve(
    env_tags: EnvTags,
    observation_space: object,
    action_space: object,
    model_spec: ModelSpec,
    *,
    trust_entrypoints: bool = False,
    check_inverse: bool = True,
) -> Adapter:
    """Derive an :class:`Adapter` for an env/model pair.

    Args:
        env_tags: The environment's observation/action tags.
        observation_space: The environment's gymnasium observation space
            (or any space object RLMesh can parse, e.g. an ``rlmesh.spaces``
            space or a native ``SpaceSpec``).
        action_space: The environment's gymnasium action space.
        model_spec: The model's declared input/output format.
        trust_entrypoints: Allow ``module:callable`` strings in custom
            inputs to be imported. Leave False for specs from untrusted
            sources; in-process callables are always allowed.
        check_inverse: Round-trip each two-armed
            :class:`~rlmesh.adapters.CustomEncoding` on a probe at resolve
            time to catch a mispaired encode/decode. Set False for an
            intentionally non-invertible encoding.

    Returns:
        Adapter applying observation preprocessing and action postprocessing.

    Raises:
        AdapterResolutionError: If a model input or action component has no
            usable counterpart in the env tags and spaces.
    """
    if check_inverse:
        _check_inverses(model_spec)
    shadow, obs_shims, act_shims = _substitute_encodings(model_spec)
    wire, customs = _model_wire(shadow, trust_entrypoints=trust_entrypoints)
    try:
        plan = adapters_resolve(
            json.dumps(env_tags.to_dict()),
            observation_space,
            action_space,
            json.dumps(wire),
        )
    except ValueError as exc:
        raise AdapterResolutionError(str(exc)) from None
    stacks = _image_stacks(shadow)
    return Adapter(plan, customs, stacks, obs_shims, act_shims)


def resolve_from_contract(
    contract: EnvContract,
    model_spec: ModelSpec,
    *,
    trust_entrypoints: bool = False,
    check_inverse: bool = True,
) -> Adapter:
    """Derive an :class:`Adapter` from an env contract and a model spec.

    Reads the env's tags from its contract metadata (published under
    :data:`ENV_METADATA_KEY` by a server set up with
    :func:`rlmesh.adapters.tag`) and its observation/action spaces from
    the contract, then resolves as in :func:`resolve`.

    Args:
        contract: The environment contract (e.g. ``remote_env.env_contract``).
        model_spec: The model's declared input/output format.
        trust_entrypoints: See :func:`resolve`.
        check_inverse: See :func:`resolve`.

    Raises:
        AdapterResolutionError: If the contract carries no tags, or
            resolution fails.
    """
    metadata = cast("dict[str, Any] | None", contract.metadata) or {}
    tags = EnvTags.from_metadata(metadata)
    if tags is None:
        raise AdapterResolutionError(
            "env contract carries no adapter tags under "
            f"{ENV_METADATA_KEY!r}; serve the env with rlmesh.adapters.tag(...)"
        )
    return resolve(
        tags,
        contract.observation_space,
        contract.action_space,
        model_spec,
        trust_entrypoints=trust_entrypoints,
        check_inverse=check_inverse,
    )


__all__ = ["resolve", "resolve_from_contract"]
