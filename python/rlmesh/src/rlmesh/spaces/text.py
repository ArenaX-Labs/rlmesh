"""Text space wrapper."""

from __future__ import annotations

from typing import cast, final

from .._rlmesh import text_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import Space
from ._internals import spec_details


@final
class Text(Space[Value]):
    """Bounded text space.

    Args:
        max_length: Maximum string length, or an existing native ``SpaceSpec``.
        min_length: Minimum string length.
        charset: Optional allowed character set.
    """

    __slots__ = ("charset", "max_length", "min_length")
    min_length: int
    max_length: int
    charset: str

    def __init__(
        self,
        max_length: int | SpaceSpec,
        min_length: int = 1,
        charset: str | None = None,
    ) -> None:
        spec = (
            max_length
            if isinstance(max_length, SpaceSpec)
            else text_space_spec(max_length, min_length, charset)
        )
        super().__init__(spec)
        details = spec_details(spec)
        self.min_length = cast(int, details["min_length"])
        self.max_length = cast(int, details["max_length"])
        self.charset = cast(str, details["charset"])
