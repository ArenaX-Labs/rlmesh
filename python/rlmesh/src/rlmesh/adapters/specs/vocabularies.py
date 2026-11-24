"""Image, resample and rotation-encoding vocabularies (typing views over native sets).

``IMAGE_LAYOUTS``, ``RESAMPLES``, ``CROP_MODES``, ``CHANNEL_ORDERS`` and
``ROTATION_DIMS`` are defined in the ``rlmesh-adapters`` crate and re-exported
through the native bindings. ``ImageLayout``/``Resample``/``CropMode``/
``ChannelOrder``/``RotationEncoding`` are typing views of the same value sets.
"""

from typing import Literal, TypeAlias

from ..._rlmesh import (
    CHANNEL_ORDERS,
    CROP_MODES,
    IMAGE_LAYOUTS,
    RESAMPLES,
    ROTATION_DIMS,
)

ImageLayout: TypeAlias = Literal["hwc", "chw"]
# ``gravity_xyz`` is observation-only: nothing converts out of it.
RotationEncoding: TypeAlias = Literal[
    "quat_xyzw",
    "quat_wxyz",
    "axis_angle",
    "rot6d",
    "rot6d_rowmajor",
    "euler_xyz",
    "gravity_xyz",
]
FitMode: TypeAlias = Literal["stretch", "crop", "pad"]

# ``_aa`` names are PIL's; bare ``bicubic``/``lanczos3`` are absent so a spec cannot silently get the other library's kernel.
Resample: TypeAlias = Literal[
    "bilinear", "bilinear_aa", "bicubic_aa", "lanczos3_aa", "area"
]

Frame: TypeAlias = Literal["world", "robot_base", "tool"]
Reference: TypeAlias = Literal["current", "target"]
Polarity: TypeAlias = Literal["open", "close"]
Provenance: TypeAlias = Literal["sensed", "estimated", "privileged"]
CropMode: TypeAlias = Literal["zoom", "slice"]
ChannelOrder: TypeAlias = Literal["rgb", "bgr"]

StackSpec: TypeAlias = tuple[int, ...]
StackPad: TypeAlias = Literal["first", "black"]

__all__ = [
    "CHANNEL_ORDERS",
    "CROP_MODES",
    "IMAGE_LAYOUTS",
    "RESAMPLES",
    "ROTATION_DIMS",
    "ChannelOrder",
    "CropMode",
    "FitMode",
    "Frame",
    "ImageLayout",
    "Polarity",
    "Provenance",
    "Reference",
    "Resample",
    "RotationEncoding",
    "StackPad",
    "StackSpec",
]
