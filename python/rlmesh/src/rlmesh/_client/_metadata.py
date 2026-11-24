from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

EMPTY_METADATA: Mapping[str, object] = MappingProxyType({})

__all__ = ["EMPTY_METADATA"]
