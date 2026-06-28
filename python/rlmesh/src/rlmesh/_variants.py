"""The catalog primitive: one concrete sub-environment a factory contains."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["Variant"]


class Variant:
    """One concrete sub-environment in an :class:`~rlmesh.EnvFactory`'s catalog.

    Return a list of these from an optional ``enumerate_variants()`` classmethod
    (or ``yield`` them lazily for a large catalog) to enumerate the finite, named
    environments a factory contains -- e.g. one per benchmark task. This is distinct from ``enumerate_params()``, which declares independent
    *sweep axes*: a catalog is a flat list of named, already-bound sub-envs, the
    right shape when the dimensions are dependent (a task index whose range depends
    on the suite) and each entry has a human identity. ``python -m rlmesh.describe``
    emits the catalog off-GPU for a managed dashboard / env hub to list and spawn.

    Args:
        id: Author-explicit, non-empty, **factory-unique** handle (e.g.
            ``"libero_10/pick_up_the_black_bowl"``). Prefer a stable upstream
            identity over a positional index -- a positional id silently repoints
            if the upstream library reorders. A hub composes a global handle as
            ``(factory identity, id)``.
        params: ``make()`` kwargs binding ONLY the identity-defining params; the
            remaining free dials stay in the ``ParamSpec`` and are composed by the
            consumer (free dials = ``param_spec`` names minus these keys). Copied
            defensively, so reusing one dict across entries in a loop is safe.
        metadata: Open display bag (keyword-only). ``name`` is the one recognized
            key a dashboard renders as the title; every other key is domain
            metadata the framework passes through untouched (e.g. a robotics env's
            ``instruction``).
    """

    __slots__ = ("id", "metadata", "params")

    def __init__(self, id: str, params: Mapping[str, Any], /, **metadata: Any) -> None:
        self.id = id
        self.params = dict(params)  # defensive copy; also rejects a non-mapping
        self.metadata = metadata
