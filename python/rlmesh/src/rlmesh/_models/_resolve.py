"""Adapter resolution from an env contract and a model spec.

``resolve_route_adapter`` is the served-route entry (resolved once per route at
``ConfigureRoute``); ``resolve_adapter`` is the shared core used by both the
served path and the local :class:`rlmesh._models._eval.Session`;
``reject_vector_env`` is the single-env guard the per-episode loop relies on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._adapter_mode import NO_ADAPTER

if TYPE_CHECKING:
    from ..adapters import Adapter
    from ..specs import EnvContract

# Cross-module surface: the public route resolver plus the helpers ``_eval``
# resolves as module globals (see note in ``_connect``).
__all__ = ["reject_vector_env", "resolve_adapter", "resolve_route_adapter"]


def resolve_route_adapter(
    spec: object | None, contract: EnvContract, trust_entrypoints: bool
) -> Adapter | None:
    """Resolve a served route's adapter from its configure-time env contract.

    The serve-path counterpart of the run(env) resolution: a served model
    receives the env contract once per route (the ``ConfigureRoute`` RPC), so it
    resolves the adapter there rather than at connect. Returns ``None`` for a
    spec-less / ``NO_ADAPTER`` model (no transform). Raises on a spec/env mismatch
    so route configuration fails loudly instead of predicting wrongly.

    Frame-stacking state is now episode-keyed in the native serving engine, so a
    stateful (frame-stacking) adapter serves correctly against a vectorized route
    -- the old single-lane rejection is lifted. (Model-*internal* state, which the
    engine cannot key by episode, is gated to single-lane by a registration-time
    probe instead.)
    """
    return resolve_adapter(spec, contract, trust_entrypoints)


def resolve_adapter(
    spec: object | None, contract: EnvContract | None, trust_entrypoints: bool
) -> Adapter | None:
    from ..adapters import (
        AdapterResolutionError,
        EnvTags,
        ModelSpec,
        resolve_from_contract,
    )

    if spec is NO_ADAPTER:
        return None
    metadata = contract.metadata if contract is not None else None
    tagged = EnvTags.from_metadata(metadata or {}) is not None
    if spec is None:
        if tagged:
            raise AdapterResolutionError(
                "the env publishes adapter tags but this model has spec=None; "
                "pass spec=<ModelSpec> to adapt, or spec=NO_ADAPTER if the model "
                "adapts its own observations"
            )
        return None
    if not isinstance(spec, ModelSpec):
        raise AdapterResolutionError(
            f"a model spec must be a ModelSpec or NO_ADAPTER; got {type(spec).__name__}"
        )
    if contract is None:
        raise AdapterResolutionError(
            "resolving a spec'd adapter requires an env contract, but the env exposes none"
        )
    return resolve_from_contract(contract, spec, trust_entrypoints=trust_entrypoints)


def reject_vector_env(contract: EnvContract | None) -> None:
    # The per-episode loop is single-env: it reads scalar reward/termination. A
    # vector env (num_envs>1) would crash on array truthiness, so reject it up
    # front rather than deep in the step loop.
    num_envs = getattr(contract, "num_envs", 1) if contract is not None else 1
    if num_envs and num_envs > 1:
        raise ValueError(
            f"Model.run() drives a single env, but the env reports num_envs={num_envs}; "
            "use num_envs=1 (the per-episode loop reads scalar reward/termination)."
        )
