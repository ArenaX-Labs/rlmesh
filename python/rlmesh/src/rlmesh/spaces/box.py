"""Box space wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, final

from .._rlmesh import box_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._internals import require_float, spec_details
from ._internals import shape as normalize_shape


@final
class Box(Space[Value]):
    """Continuous box space.

    Args:
        low: Lower bound, or an existing native ``SpaceSpec``.
        high: Upper bound when constructing a new spec.
        shape: Box shape when constructing a new spec.
        dtype: Element dtype name.
    """

    __slots__ = ("bounds_kind", "high", "low")
    bounds_kind: str | None
    low: object
    high: object

    def __init__(
        self,
        low: float | SpaceSpec,
        high: float | None = None,
        shape: Sequence[int] | None = None,
        dtype: str = "float32",
    ) -> None:
        spec = (
            low
            if isinstance(low, SpaceSpec)
            else box_space_spec(
                float(low), require_float(high, "high"), normalize_shape(shape), dtype
            )
        )
        super().__init__(spec)
        details = spec_details(spec)
        self.bounds_kind = cast(str | None, details.get("bounds_kind"))
        self.low = details.get("low")
        self.high = details.get("high")
