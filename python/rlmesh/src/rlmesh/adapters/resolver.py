"""Resolve env and model IO specs into a concrete adapter.

Resolution and plan application run in the native ``rlmesh-adapters``
core (the same implementation behind every language binding); this module
serializes the specs across the boundary and keeps the host-language
concerns where they belong: entrypoint trust gating, custom callables,
and the error type.
"""

from __future__ import annotations

import json
from typing import Any, cast

from .._bootstrap.entrypoint import resolve_entrypoint
from .._rlmesh import adapters_resolve
from .adapter import IOAdapter
from .errors import AdapterResolutionError
from .specs import CustomInput, EnvIOSpec, ModelIOSpec, ObsTransform
from .specs.action_serialization import action_layout_to_dict
from .specs.model_serialization import model_input_to_dict


def resolve(
    env_spec: EnvIOSpec,
    model_spec: ModelIOSpec,
    *,
    trust_entrypoints: bool = False,
) -> IOAdapter:
    """Derive an :class:`IOAdapter` for an env/model pair from their specs.

    Args:
        env_spec: The environment's declared observation/action format.
        model_spec: The model's declared input/output format.
        trust_entrypoints: Allow ``module:callable`` strings in custom
            inputs to be imported. Leave False for specs from untrusted
            sources; in-process callables are always allowed.

    Returns:
        Adapter applying observation preprocessing and action postprocessing.

    Raises:
        AdapterResolutionError: If a model input or action component has no
            usable counterpart in the env spec.
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
    model_wire = {
        "inputs": wire_inputs,
        "action": action_layout_to_dict(model_spec.action),
    }
    try:
        plan = adapters_resolve(json.dumps(env_spec.to_dict()), json.dumps(model_wire))
    except ValueError as exc:
        raise AdapterResolutionError(str(exc)) from None
    return IOAdapter(plan, customs)


__all__ = ["resolve"]
