"""The ``tag`` verb: attach env tags to an environment.

Tagging an environment publishes its :class:`EnvTags` in the
env's ``metadata`` mapping (under :data:`ENV_METADATA_KEY`). A server built
on that env then carries the tags in its contract, so a model client
can resolve an adapter from the handshake alone
(:func:`rlmesh.adapters.resolve_from_contract`) without a local copy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from .._rlmesh import adapters_join_check
from .errors import AdapterResolutionError
from .specs import EnvTags

EnvT = TypeVar("EnvT")


def tag(env: EnvT, tags: EnvTags, *, validate: bool = True) -> EnvT:
    """Attach env tags to ``env`` and return it (for chaining).

    The tags are merged into ``env.metadata`` under
    :data:`ENV_METADATA_KEY`, leaving any existing metadata intact.

    Args:
        env: The environment to tag (a gymnasium-style env exposing
            ``metadata`` and, for validation, ``observation_space`` and
            ``action_space``).
        tags: The observation/action tags to publish.
        validate: When True (default), check the tags against the
            env's observation and action spaces via the native ``join``,
            failing fast if a role or width cannot be reconciled.

    Returns:
        The same ``env`` object, now carrying the tags.

    Raises:
        AdapterResolutionError: If ``validate`` is set and the tags
            do not reconcile with the env's spaces.
    """
    if validate:
        _validate(env, tags)
    existing = getattr(env, "metadata", None)
    merged: dict[str, Any] = (
        dict(cast("Mapping[str, Any]", existing))
        if isinstance(existing, Mapping)
        else {}
    )
    merged.update(tags.to_metadata())
    # ``metadata`` is typically a class attribute on gymnasium envs; assigning
    # here shadows it with an instance attribute, the standard override path.
    env.metadata = merged  # type: ignore[attr-defined]
    return env


def _validate(env: object, tags: EnvTags) -> None:
    observation_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)
    if observation_space is None or action_space is None:
        raise AdapterResolutionError(
            "cannot validate tags: env exposes no observation_space/"
            "action_space; pass validate=False to tag without checking"
        )
    try:
        adapters_join_check(json.dumps(tags.to_dict()), observation_space, action_space)
    except ValueError as exc:
        raise AdapterResolutionError(str(exc)) from None


__all__ = ["tag"]
