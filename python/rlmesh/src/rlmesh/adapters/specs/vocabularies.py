"""Image-layout and rotation-encoding vocabularies (typing views over native sets).

``IMAGE_LAYOUTS`` and ``ROTATION_DIMS`` are each defined once, in the
``rlmesh-adapters`` crate (``ImageLayout::ALL`` in ``v1/spec/layouts.rs`` and
``RotationEncoding::dims`` in ``v1/spec/rotations.rs``); this module re-exports
them through the native bindings. ``ImageLayout``/``RotationEncoding`` are the
Python-side typing views of the same value sets.
"""

from typing import Literal, TypeAlias

from ..._rlmesh import IMAGE_LAYOUTS, ROTATION_DIMS

ImageLayout: TypeAlias = Literal["hwc", "chw"]
RotationEncoding: TypeAlias = Literal[
    "quat_xyzw", "quat_wxyz", "axis_angle", "rot6d", "rot6d_rowmajor", "euler_xyz"
]
# Typing view of the frozen FitMode vocab (FitMode::ALL in v1/spec/layouts.rs).
# Validated by the Rust codec at normalize; this only gives authors static
# checking. No native FIT_MODES export yet (unlike IMAGE_LAYOUTS/ROTATION_DIMS).
FitMode: TypeAlias = Literal["stretch", "crop", "pad"]

__all__ = [
    "IMAGE_LAYOUTS",
    "ROTATION_DIMS",
    "FitMode",
    "ImageLayout",
    "RotationEncoding",
]
