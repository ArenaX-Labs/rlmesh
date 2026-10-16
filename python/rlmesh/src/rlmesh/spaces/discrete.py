"""Discrete space wrapper."""

from __future__ import annotations

from typing import cast, final

from .._rlmesh import discrete_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._internals import dtype_name, require_spec_kind, spec_details
from ._literals import DiscreteDTypeLike


@final
class Discrete(Space[Value]):
    """Discrete integer space.

    Args:
        n: Number of values, or an existing native ``SpaceSpec``.
        start: First integer value in the space.
        dtype: Integer dtype name: ``"int64"``, ``"int32"`` or ``"uint8"``.
    """

    __slots__ = ("n", "start")
    n: int
    start: int

    def __init__(
        self, n: int | SpaceSpec, start: int = 0, dtype: DiscreteDTypeLike = "int64"
    ) -> None:
        spec = (
            require_spec_kind(n, "discrete")
            if isinstance(n, SpaceSpec)
            else discrete_space_spec(n, start, dtype_name(dtype))
        )
        super().__init__(spec)
        details = spec_details(spec)
        self.n = cast(int, details["n"])
        self.start = cast(int, details["start"])
