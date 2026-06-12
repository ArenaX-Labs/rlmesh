"""Observation feature dataclasses declared by environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from ..constants import IMAGE_PRIMARY, INSTRUCTION, JOINT_POS
from .layouts import ImageLayout
from .rotations import RotationEncoding


@dataclass(frozen=True)
class EnvImage:
    """A camera image entry in an environment observation.

    Attributes:
        key: Observation dict key (dotted paths traverse nested dicts).
        role: Semantic role used for matching, e.g. ``image/primary``.
        layout: Axis layout of the stored image.
        upside_down: Whether the image is rendered rotated 180 degrees
            relative to the canonical upright orientation.
    """

    key: str
    role: str = IMAGE_PRIMARY
    layout: ImageLayout = "hwc"
    upside_down: bool = False


@dataclass(frozen=True)
class EnvState:
    """A numeric proprioception entry in an environment observation.

    Attributes:
        key: Observation dict key (dotted paths traverse nested dicts).
        role: Semantic role used for matching, e.g. ``proprio/eef_pos``.
        dim: Optional flattened length, for documentation and validation.
        encoding: Rotation encoding when the role is a rotation.
        range: Optional ``(low, high)`` value range of the feature.
    """

    key: str
    role: str = JOINT_POS
    dim: int | None = None
    encoding: RotationEncoding | None = None
    range: tuple[float, float] | None = None


@dataclass(frozen=True)
class EnvText:
    """A text entry (typically the task instruction) in an observation.

    Attributes:
        key: Observation dict key.
        role: Semantic role used for matching.
    """

    key: str
    role: str = INSTRUCTION


EnvFeature: TypeAlias = EnvImage | EnvState | EnvText

__all__ = ["EnvFeature", "EnvImage", "EnvState", "EnvText"]
