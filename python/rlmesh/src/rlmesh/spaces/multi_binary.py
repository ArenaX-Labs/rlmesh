"""MultiBinary space wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, final

from .._rlmesh import multi_binary_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._utils import shape as space_shape
from ._utils import spec_details


@final
class MultiBinary(Space[Value]):
    """Binary vector or tensor space.

    Args:
        shape: Number of binary values, tensor shape, or native ``SpaceSpec``.
    """

    __slots__ = ("dims", "size")
    size: int | None
    dims: list[int] | None

    def __init__(self, shape: int | Sequence[int] | SpaceSpec) -> None:
        spec = (
            shape
            if isinstance(shape, SpaceSpec)
            else multi_binary_space_spec(
                space_shape([shape] if isinstance(shape, int) else shape)
            )
        )
        super().__init__(spec)
        details = spec_details(spec)
        self.size = cast(int | None, details.get("size"))
        self.dims = cast(list[int] | None, details.get("dims"))
