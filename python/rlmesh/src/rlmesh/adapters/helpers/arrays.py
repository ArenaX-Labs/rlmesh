"""NumPy availability guard and array type alias for the adapters package."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    NumpyArray: TypeAlias = np.ndarray[Any, Any]
else:
    NumpyArray: TypeAlias = object


def ensure_available() -> None:
    """Raise if NumPy is not installed."""
    try:
        _ = importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "rlmesh.adapters requires numpy. Install rlmesh[numpy]."
        ) from exc


__all__ = ["NumpyArray", "ensure_available"]
