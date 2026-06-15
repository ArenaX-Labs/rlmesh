"""Resolve env tags and a model spec into a concrete adapter.

Resolution and plan application run in the native ``rlmesh-adapters`` core
(the same implementation behind every language binding); this module
serializes the tags and spec across the boundary, hands the
gymnasium observation/action spaces to the native projector, and keeps the
host-language concerns where they belong: entrypoint trust gating, custom
callables, and the error type.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from .._entrypoint import resolve_entrypoint
from .._rlmesh import ROTATION_DIMS, adapters_resolve
from .adapter import ActEncShim, Adapter, ObsEncShim
from .constants import ENV_METADATA_KEY
from .specs import (
    ActionComponent,
    ActionLayout,
    CustomEncoding,
    EntrypointCustomInput,
    EnvTags,
    ImageInput,
    InlineCustomInput,
    ModelSpec,
    ObsTransform,
    RotationTransform,
    StateInput,
)
from .specs.action_serialization import action_layout_to_dict
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


def _shadow_state_input(
    state_input: StateInput, obs_shims: list[ObsEncShim]
) -> StateInput:
    """Replace a custom-encoded state input with its base-encoding shadow.

    Records an observation shim. A custom encoding must be the sole component
    of a single-piece input, with no width-altering or assembly options.
    """
    for component in state_input.components:
        _check_native_encoding(component.encoding, f"state input {state_input.key!r}")
    custom = [
        c for c in state_input.components if isinstance(c.encoding, CustomEncoding)
    ]
    if not custom:
        return state_input
    key = state_input.key
    if len(state_input.components) != 1:
        raise AdapterResolutionError(
            f"state input {key!r} uses a CustomEncoding, which must be the sole "
            "component of a single-piece StateInput (observation offsets are "
            "env-dependent); give the rotation its own input key"
        )
    component = state_input.components[0]
    encoding = cast(CustomEncoding, component.encoding)
    if component.dim is not None or component.index is not None:
        raise AdapterResolutionError(
            f"state input {key!r}: a CustomEncoding component cannot also set "
            "dim or index (they would change its width)"
        )
    if component.optional:
        raise AdapterResolutionError(
            f"state input {key!r}: a CustomEncoding component cannot be optional"
        )
    if state_input.pad_to is not None or state_input.reshape is not None:
        raise AdapterResolutionError(
            f"state input {key!r}: pad_to/reshape run before the host-side "
            "encoding shim and would break it; drop them"
        )
    if state_input.container != "array":
        raise AdapterResolutionError(
            f"state input {key!r}: a CustomEncoding requires container='array'"
        )
    if encoding.is_entrypoint:
        raise AdapterResolutionError(
            f"state input {key!r}: entrypoint CustomEncoding is not yet "
            "supported; use in-process callables for now"
        )
    if encoding.from_base is None:
        raise AdapterResolutionError(
            f"state input {key!r}: an observation CustomEncoding needs from_base"
        )
    obs_shims.append(
        ObsEncShim(
            model_key=key,
            base=encoding.base,
            width=encoding.width,
            dtype=state_input.dtype,
            name=encoding.name,
            from_base=cast("RotationTransform", encoding.from_base),
        )
    )
    shadow_component = replace(component, encoding=encoding.base)
    return replace(state_input, components=(shadow_component,))


def _shadow_action(action: ActionLayout, act_shims: list[ActEncShim]) -> ActionLayout:
    """Replace custom-encoded action components with base-encoding shadows.

    Records action shims with model-declared offsets.
    """
    shadow_components: list[ActionComponent] = []
    offset = 0
    for component in action.components:
        _check_native_encoding(
            component.encoding, f"action component {component.role!r}"
        )
        if isinstance(component.encoding, CustomEncoding):
            encoding = component.encoding
            if component.binary:
                raise AdapterResolutionError(
                    f"action component {component.role!r}: a CustomEncoding "
                    "cannot be binary"
                )
            if encoding.is_entrypoint:
                raise AdapterResolutionError(
                    f"action component {component.role!r}: entrypoint "
                    "CustomEncoding is not yet supported; use in-process "
                    "callables for now"
                )
            if encoding.to_base is None:
                raise AdapterResolutionError(
                    f"action component {component.role!r}: an action "
                    "CustomEncoding needs to_base"
                )
            act_shims.append(
                ActEncShim(
                    offset=offset,
                    width=component.dim,
                    base=encoding.base,
                    name=encoding.name,
                    to_base=cast("RotationTransform", encoding.to_base),
                )
            )
            shadow_components.append(replace(component, encoding=encoding.base))
        else:
            shadow_components.append(component)
        offset += component.dim
    return ActionLayout(*shadow_components, clip=action.clip)


def _custom_encodings(model_spec: ModelSpec) -> Iterator[CustomEncoding]:
    for model_input in model_spec.inputs:
        if isinstance(model_input, StateInput):
            for component in model_input.components:
                if isinstance(component.encoding, CustomEncoding):
                    yield component.encoding
    for component in model_spec.action.components:
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
    repack the custom fields at the boundary.
    """
    obs_shims: list[ObsEncShim] = []
    act_shims: list[ActEncShim] = []
    shadow_inputs = tuple(
        _shadow_state_input(model_input, obs_shims)
        if isinstance(model_input, StateInput)
        else model_input
        for model_input in model_spec.inputs
    )
    shadow_action = _shadow_action(model_spec.action, act_shims)
    shadow = ModelSpec(inputs=shadow_inputs, action=shadow_action)
    return shadow, tuple(obs_shims), tuple(act_shims)


def _model_wire(
    model_spec: ModelSpec, *, trust_entrypoints: bool
) -> tuple[dict[str, Any], dict[str, ObsTransform]]:
    """Split a model spec into its wire form and the local custom callables.

    Custom inputs never cross to the native core as code: an entrypoint
    string is gated on ``trust_entrypoints`` and imported here; an
    in-process callable stays here and is referenced by a ``host:<key>``
    placeholder. Either way the native plan keeps the input as a hole that
    :class:`Adapter` fills from the raw observation.
    """
    customs: dict[str, ObsTransform] = {}
    wire_inputs: list[dict[str, Any]] = []
    for model_input in model_spec.inputs:
        if isinstance(model_input, InlineCustomInput):
            customs[model_input.key] = model_input.transform
            wire_transform = f"host:{model_input.key}"
        elif isinstance(model_input, EntrypointCustomInput):
            if not trust_entrypoints:
                raise AdapterResolutionError(
                    f"custom input {model_input.key!r} references entrypoint "
                    f"{model_input.entrypoint!r}; pass "
                    "resolve(..., trust_entrypoints=True) to allow importing it"
                )
            customs[model_input.key] = cast(
                ObsTransform,
                resolve_entrypoint(
                    model_input.entrypoint, label="custom input transform"
                ),
            )
            wire_transform = model_input.entrypoint
        else:
            wire_inputs.append(model_input_to_dict(model_input))
            continue
        wire_inputs.append(
            {"type": "custom", "key": model_input.key, "transform": wire_transform}
        )
    wire = {"inputs": wire_inputs, "action": action_layout_to_dict(model_spec.action)}
    return wire, customs


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
    stacks = {
        model_input.key: model_input.stack
        for model_input in shadow.inputs
        if isinstance(model_input, ImageInput) and model_input.stack > 1
    }
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
