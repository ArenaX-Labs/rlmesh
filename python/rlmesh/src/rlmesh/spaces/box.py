"""Box space wrapper."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, final

from .._rlmesh import box_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._internals import dtype_name, require_float, spec_details
from ._internals import shape as normalize_shape
from ._literals import Bound, FloatDTypeLike


def _bound(value: Bound | None, name: str) -> float:
    """Validate one Box bound, keeping an exact Python ``int`` exact.

    ``float()`` rounds integer bounds above 2**53 and saturates a ``uint64``
    bound at ``i64::MAX``; the native constructor routes exact integers through
    the typed builders instead, so only non-integers need the float coercion
    (which is also where ``"inf"``/``"-inf"`` parse and NaN is rejected).
    """
    if isinstance(value, int):
        return value
    return require_float(value, name)


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
        low: Bound | SpaceSpec,
        high: Bound | None = None,
        shape: Sequence[int] | None = None,
        dtype: FloatDTypeLike = "float32",
    ) -> None:
        spec = (
            low
            if isinstance(low, SpaceSpec)
            else box_space_spec(
                _bound(low, "low"),
                _bound(high, "high"),
                normalize_shape(shape),
                dtype_name(dtype),
            )
        )
        super().__init__(spec)
        details = spec_details(spec)
        self.bounds_kind = cast(str | None, details.get("bounds_kind"))
        self.low = details.get("low")
        self.high = details.get("high")
