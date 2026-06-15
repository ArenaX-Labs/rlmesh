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

__all__ = ["IMAGE_LAYOUTS", "ROTATION_DIMS", "ImageLayout", "RotationEncoding"]
