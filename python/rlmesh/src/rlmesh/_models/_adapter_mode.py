"""Model adapter-mode sentinels."""

from __future__ import annotations


class _NoAdapter:
    """Sentinel: the model handles raw env observations/actions itself."""

    def __repr__(self) -> str:
        return "NO_ADAPTER"


NO_ADAPTER = _NoAdapter()
"""Pass as ``spec`` to explicitly skip RLMesh adapter resolution."""


__all__ = ["NO_ADAPTER"]
