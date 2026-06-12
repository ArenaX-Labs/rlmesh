"""Named RLMesh-native spaces and optional Gymnasium conversions."""

from __future__ import annotations

from ..specs import SpaceSpec
from ._base import Space, SpaceBridge
from ._conversion import from_gymnasium_space, to_gymnasium_space
from ._registry import space_from_spec
from .box import Box
from .dict import Dict
from .discrete import Discrete
from .multi_binary import MultiBinary
from .multi_discrete import MultiDiscrete
from .text import Text
from .tuple import Tuple

__all__ = [
    "Box",
    "Dict",
    "Discrete",
    "MultiBinary",
    "MultiDiscrete",
    "Space",
    "SpaceBridge",
    "SpaceSpec",
    "Text",
    "Tuple",
    "from_gymnasium_space",
    "space_from_spec",
    "to_gymnasium_space",
]
