"""The ``tag`` verb: attach env tags to an environment.

Tagging an environment publishes its :class:`EnvTags` in the
env's ``metadata`` mapping (under :data:`ENV_METADATA_KEY`). A server built
on that env then carries the tags in its contract, so a model client
can resolve an adapter from the handshake alone
(:func:`rlmesh.adapters.resolve_from_contract`) without a local copy.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from typing import Any, TypeVar, cast

from .._rlmesh import adapters_join_check
from .resolver import AdapterResolutionError
from .specs import EnvTags

EnvT = TypeVar("EnvT")


def tag(env: EnvT, tags: EnvTags, *, validate: bool = True) -> EnvT:
    """Merge env tags into ``env.metadata`` (under :data:`ENV_METADATA_KEY`) and return it.

    With ``validate=True`` (default), check the tags against the env's
    observation/action spaces via the native ``join``, raising
    :class:`AdapterResolutionError` if a role or width cannot be reconciled.
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
        advisories = adapters_join_check(
            json.dumps(tags.to_dict()), observation_space, action_space
        )
    except ValueError as exc:
        raise AdapterResolutionError(str(exc)) from None
    # Non-fatal hints (e.g. a layout that looks mis-declared): surface at tag time
    # so the author sees their own mistake now, not via a peer's serve logs.
    for note in advisories:
        warnings.warn(note, stacklevel=3)


__all__ = ["tag"]
