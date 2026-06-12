"""Tuple space wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast, final

from .._rlmesh import tuple_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import NewOutputT, Space, SpaceBridge
from ._utils import spec_details


@final
class Tuple(Space[Value]):
    """Ordered tuple of child spaces.

    Args:
        spaces: Iterable of child spaces, or an existing native ``SpaceSpec``.
    """

    __slots__ = ("spaces",)
    spaces: tuple[Space[Value], ...]

    def __init__(self, spaces: Iterable[Space[Any] | SpaceSpec] | SpaceSpec) -> None:
        if isinstance(spaces, SpaceSpec):
            if spaces.kind != "tuple":
                raise ValueError(f"expected tuple SpaceSpec, got {spaces.kind!r}")
            spec = spaces
        else:
            entries: list[object] = list(spaces)
            spec = tuple_space_spec(entries)
        super().__init__(spec)
        details = spec_details(spec)
        raw_spaces = cast(list[SpaceSpec], details.get("spaces", []))
        self.spaces = tuple(_space_from_spec(child) for child in raw_spaces)

    def _with_bridge(self, bridge: SpaceBridge[NewOutputT]) -> Space[NewOutputT]:
        super()._with_bridge(bridge)
        self.spaces = tuple(
            cast(Space[Value], child._with_bridge(bridge)) for child in self.spaces
        )
        return cast(Space[NewOutputT], self)


def _space_from_spec(spec: SpaceSpec) -> Space[Value]:
    from ._registry import space_from_spec

    return space_from_spec(spec)
