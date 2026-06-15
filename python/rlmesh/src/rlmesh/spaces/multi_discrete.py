"""MultiDiscrete space wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, final

from .._rlmesh import multi_discrete_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._internals import spec_details


@final
class MultiDiscrete(Space[Value]):
    """Vector of discrete dimensions.

    Args:
        nvec: Per-dimension counts, or an existing native ``SpaceSpec``.
        dtype: Integer dtype name.
    """

    __slots__ = ("nvec",)
    nvec: list[int] | None

    def __init__(self, nvec: Sequence[int] | SpaceSpec, dtype: str = "int64") -> None:
        spec = (
            nvec
            if isinstance(nvec, SpaceSpec)
            else multi_discrete_space_spec(list(nvec), dtype)
        )
        super().__init__(spec)
        details = spec_details(spec)
        self.nvec = cast(list[int] | None, details.get("nvec"))
