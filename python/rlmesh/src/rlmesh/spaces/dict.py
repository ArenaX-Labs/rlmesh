"""Dict space wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast, final

from .._rlmesh import dict_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import NewOutputT, Space, SpaceAdapter
from ._utils import EMPTY_SPACE_MAPPING, spec_details


@final
class Dict(Space[Value]):
    """Mapping of named child spaces.

    Args:
        spaces: Mapping of child spaces, or an existing native ``SpaceSpec``.
    """

    __slots__ = ("spaces",)
    spaces: Mapping[str, Space[Value]]

    def __init__(
        self, spaces: Mapping[str, Space[Any] | SpaceSpec] | SpaceSpec
    ) -> None:
        if isinstance(spaces, SpaceSpec):
            if spaces.kind != "dict":
                raise ValueError(f"expected dict SpaceSpec, got {spaces.kind!r}")
            spec = spaces
        else:
            entries: dict[str, object] = {key: child for key, child in spaces.items()}
            spec = dict_space_spec(entries)
        super().__init__(spec)
        details = spec_details(spec)
        raw_spaces = cast(
            Mapping[str, SpaceSpec], details.get("spaces", EMPTY_SPACE_MAPPING)
        )
        self.spaces = MappingProxyType(
            {key: _space_from_spec(child) for key, child in raw_spaces.items()}
        )

    def _with_adapter(self, adapter: SpaceAdapter[NewOutputT]) -> Space[NewOutputT]:
        super()._with_adapter(adapter)
        self.spaces = MappingProxyType(
            {
                key: cast(Space[Value], child._with_adapter(adapter))
                for key, child in self.spaces.items()
            }
        )
        return cast(Space[NewOutputT], self)


def _space_from_spec(spec: SpaceSpec) -> Space[Value]:
    from ._registry import space_from_spec

    return space_from_spec(spec)
