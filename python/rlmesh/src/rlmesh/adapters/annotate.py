"""The ``annotate`` verb: attach env IO annotations to an environment.

Annotating an environment publishes its :class:`EnvAnnotations` in the
env's ``metadata`` mapping (under :data:`ENV_METADATA_KEY`). A server built
on that env then carries the annotations in its contract, so a model client
can resolve an adapter from the handshake alone
(:func:`rlmesh.adapters.resolve_from_contract`) without a local copy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from .._rlmesh import adapters_join_check
from .errors import AdapterResolutionError
from .specs import EnvAnnotations

EnvT = TypeVar("EnvT")


def annotate(env: EnvT, annotations: EnvAnnotations, *, validate: bool = True) -> EnvT:
    """Attach IO annotations to ``env`` and return it (for chaining).

    The annotations are merged into ``env.metadata`` under
    :data:`ENV_METADATA_KEY`, leaving any existing metadata intact.

    Args:
        env: The environment to annotate (a gymnasium-style env exposing
            ``metadata`` and, for validation, ``observation_space`` and
            ``action_space``).
        annotations: The observation/action annotations to publish.
        validate: When True (default), check the annotations against the
            env's observation and action spaces via the native ``join``,
            failing fast if a role or width cannot be reconciled.

    Returns:
        The same ``env`` object, now carrying the annotations.

    Raises:
        AdapterResolutionError: If ``validate`` is set and the annotations
            do not reconcile with the env's spaces.
    """
    if validate:
        _validate(env, annotations)
    existing = getattr(env, "metadata", None)
    merged: dict[str, Any] = (
        dict(cast("Mapping[str, Any]", existing))
        if isinstance(existing, Mapping)
        else {}
    )
    merged.update(annotations.to_metadata())
    # ``metadata`` is typically a class attribute on gymnasium envs; assigning
    # here shadows it with an instance attribute, the standard override path.
    env.metadata = merged  # type: ignore[attr-defined]
    return env


def _validate(env: object, annotations: EnvAnnotations) -> None:
    observation_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)
    if observation_space is None or action_space is None:
        raise AdapterResolutionError(
            "cannot validate annotations: env exposes no observation_space/"
            "action_space; pass validate=False to annotate without checking"
        )
    try:
        adapters_join_check(
            json.dumps(annotations.to_dict()), observation_space, action_space
        )
    except ValueError as exc:
        raise AdapterResolutionError(str(exc)) from None


__all__ = ["annotate"]
