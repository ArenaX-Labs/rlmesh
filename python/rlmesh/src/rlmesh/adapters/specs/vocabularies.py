"""Image, resample and rotation-encoding vocabularies (typing views over native sets).

``IMAGE_LAYOUTS``, ``RESAMPLES`` and ``ROTATION_DIMS`` are each defined once, in
the ``rlmesh-adapters`` crate (``ImageLayout::ALL`` in ``v1/spec/layouts.rs``,
``RESAMPLES`` in ``v1/apply/image.rs``, and ``RotationEncoding::dims`` in
``v1/spec/rotations.rs``); this module re-exports them through the native
bindings. ``ImageLayout``/``Resample``/``RotationEncoding`` are the Python-side
typing views of the same value sets.
"""

from typing import Literal, TypeAlias

from ..._rlmesh import IMAGE_LAYOUTS, RESAMPLES, ROTATION_DIMS

ImageLayout: TypeAlias = Literal["hwc", "chw"]
RotationEncoding: TypeAlias = Literal[
    "quat_xyzw", "quat_wxyz", "axis_angle", "rot6d", "rot6d_rowmajor", "euler_xyz"
]
# Typing view of the frozen FitMode vocab (FitMode::ALL in v1/spec/layouts.rs).
# Validated by the Rust codec at normalize; this only gives authors static
# checking. No native FIT_MODES export yet (unlike IMAGE_LAYOUTS/ROTATION_DIMS).
FitMode: TypeAlias = Literal["stretch", "crop", "pad"]

# Un-suffixed names are cv2/torch semantics, ``_aa`` names are PIL's (the
# filter support widens with the downscale factor). Bare ``bicubic``/``lanczos3``
# are deliberately absent: a spec naming one fails resolution rather than
# silently getting the other library's kernel.
Resample: TypeAlias = Literal[
    "bilinear", "bilinear_aa", "bicubic_aa", "lanczos3_aa", "area"
]

# The geometry vocabularies (FRAMES / REFERENCES in v1/spec/frames.rs), typed
# here for static checking and validated by the Rust codec at normalize. `Frame`
# qualifies an absolute pose -- the coordinate frame its numbers are in;
# `Reference` qualifies a delta -- the pose the controller integrates it against.
Frame: TypeAlias = Literal["world", "robot_base"]
Reference: TypeAlias = Literal["current", "target"]

__all__ = [
    "IMAGE_LAYOUTS",
    "RESAMPLES",
    "ROTATION_DIMS",
    "FitMode",
    "Frame",
    "ImageLayout",
    "Reference",
    "Resample",
    "RotationEncoding",
]
