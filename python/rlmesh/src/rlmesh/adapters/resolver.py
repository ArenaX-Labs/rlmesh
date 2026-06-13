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
from typing import TYPE_CHECKING, Any, cast

from .._bootstrap.entrypoint import resolve_entrypoint
from .._rlmesh import adapters_resolve
from .adapter import IOAdapter
from .constants import ENV_METADATA_KEY
from .errors import AdapterResolutionError
from .specs import CustomInput, EnvTags, ModelSpec, ObsTransform
from .specs.action_serialization import action_layout_to_dict
from .specs.model_serialization import model_input_to_dict

if TYPE_CHECKING:
    from ..specs import EnvContract


def _model_wire(
    model_spec: ModelSpec, *, trust_entrypoints: bool
) -> tuple[dict[str, Any], dict[str, ObsTransform]]:
    """Split a model spec into its wire form and the local custom callables.

    Custom inputs never cross to the native core as code: an entrypoint
    string is gated on ``trust_entrypoints`` and imported here; an
    in-process callable stays here and is referenced by a ``host:<key>``
    placeholder. Either way the native plan keeps the input as a hole that
    :class:`IOAdapter` fills from the raw observation.
    """
    customs: dict[str, ObsTransform] = {}
    wire_inputs: list[dict[str, Any]] = []
    for model_input in model_spec.inputs:
        if not isinstance(model_input, CustomInput):
            wire_inputs.append(model_input_to_dict(model_input))
            continue
        transform = model_input.transform
        if isinstance(transform, str):
            if not trust_entrypoints:
                raise AdapterResolutionError(
                    f"custom input {model_input.key!r} references entrypoint "
                    f"{transform!r}; pass resolve(..., trust_entrypoints=True) "
                    "to allow importing it"
                )
            customs[model_input.key] = cast(
                ObsTransform,
                resolve_entrypoint(transform, label="custom input transform"),
            )
            wire_transform = transform
        else:
            customs[model_input.key] = transform
            wire_transform = f"host:{model_input.key}"
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
) -> IOAdapter:
    """Derive an :class:`IOAdapter` for an env/model pair.

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

    Returns:
        Adapter applying observation preprocessing and action postprocessing.

    Raises:
        AdapterResolutionError: If a model input or action component has no
            usable counterpart in the env tags and spaces.
    """
    wire, customs = _model_wire(model_spec, trust_entrypoints=trust_entrypoints)
    try:
        plan = adapters_resolve(
            json.dumps(env_tags.to_dict()),
            observation_space,
            action_space,
            json.dumps(wire),
        )
    except ValueError as exc:
        raise AdapterResolutionError(str(exc)) from None
    return IOAdapter(plan, customs)


def resolve_from_contract(
    contract: EnvContract,
    model_spec: ModelSpec,
    *,
    trust_entrypoints: bool = False,
) -> IOAdapter:
    """Derive an :class:`IOAdapter` from an env contract and a model spec.

    Reads the env's tags from its contract metadata (published under
    :data:`ENV_METADATA_KEY` by a server set up with
    :func:`rlmesh.adapters.tag`) and its observation/action spaces from
    the contract, then resolves as in :func:`resolve`.

    Args:
        contract: The environment contract (e.g. ``remote_env.env_contract``).
        model_spec: The model's declared input/output format.
        trust_entrypoints: See :func:`resolve`.

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
    )


__all__ = ["resolve", "resolve_from_contract"]
