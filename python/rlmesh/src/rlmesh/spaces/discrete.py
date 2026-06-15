"""Discrete space wrapper."""

from __future__ import annotations

from typing import cast, final

from .._rlmesh import discrete_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._internals import spec_details


@final
class Discrete(Space[Value]):
    """Discrete integer space.

    Args:
        n: Number of values, or an existing native ``SpaceSpec``.
        start: First integer value in the space.
        dtype: Integer dtype name.
    """

    __slots__ = ("n", "start")
    n: int
    start: int

    def __init__(
        self, n: int | SpaceSpec, start: int = 0, dtype: str = "int64"
    ) -> None:
        spec = n if isinstance(n, SpaceSpec) else discrete_space_spec(n, start, dtype)
        super().__init__(spec)
        details = spec_details(spec)
        self.n = cast(int, details["n"])
        self.start = cast(int, details["start"])
