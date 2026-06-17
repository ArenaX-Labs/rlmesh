"""Dict space wrapper."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from types import MappingProxyType
from typing import Any, cast, final

from .._rlmesh import dict_space_spec
from ..specs import SpaceSpec
from ..types import Value
from ._base import NewOutputT, Space, SpaceBridge
from ._internals import EMPTY_SPACE_MAPPING, spec_details


@final
class Dict(Space[Value]):
    """Mapping of named child spaces.

    Exposes read-only mapping access over its children, mirroring
    ``gymnasium.spaces.Dict``: index a child by name (``space["obs"]``), iterate
    names (``for name in space``), test membership (``"obs" in space``), take
    ``len(space)``, and read ``keys()`` / ``values()`` / ``items()``. The
    children are immutable (a ``MappingProxyType``); build a new ``Dict`` to
    change them.

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

    def keys(self) -> KeysView[str]:
        """The subspace names."""
        return self.spaces.keys()

    def values(self) -> ValuesView[Space[Value]]:
        """The subspaces, in key order."""
        return self.spaces.values()

    def items(self) -> ItemsView[str, Space[Value]]:
        """The ``(name, subspace)`` pairs."""
        return self.spaces.items()

    def __getitem__(self, key: str) -> Space[Value]:
        return self.spaces[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.spaces)

    def __len__(self) -> int:
        return len(self.spaces)

    def __contains__(self, key: object) -> bool:
        return key in self.spaces

    def _with_bridge(self, bridge: SpaceBridge[NewOutputT]) -> Space[NewOutputT]:
        super()._with_bridge(bridge)
        self.spaces = MappingProxyType(
            {
                key: cast(Space[Value], child._with_bridge(bridge))
                for key, child in self.spaces.items()
            }
        )
        return cast(Space[NewOutputT], self)


def _space_from_spec(spec: SpaceSpec) -> Space[Value]:
    from ._registry import space_from_spec

    return space_from_spec(spec)
