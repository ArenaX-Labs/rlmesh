"""Shared empty-metadata sentinel for RLMesh remote clients."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# Immutable empty mapping returned when an endpoint reports no metadata, so
# callers always see a read-only ``Mapping`` instead of ``None``.
EMPTY_METADATA: Mapping[str, object] = MappingProxyType({})

__all__ = ["EMPTY_METADATA"]
