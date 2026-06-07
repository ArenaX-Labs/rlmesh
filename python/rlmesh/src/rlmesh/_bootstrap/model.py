"""Private model loaders shared by Python model entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from rlmesh._bootstrap.entrypoint import parse_entrypoint, resolve_entrypoint

if TYPE_CHECKING:
    from rlmesh.numpy import NumpyValue


def load_predict(entrypoint: str) -> Callable[[NumpyValue], NumpyValue]:
    """Load a model prediction callable from ``module:callable`` syntax."""
    value = resolve_entrypoint(entrypoint, label="model entrypoint")
    return cast(Callable[[Any], Any], value)


__all__ = ["load_predict", "parse_entrypoint"]
