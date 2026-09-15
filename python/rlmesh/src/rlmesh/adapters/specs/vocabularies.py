"""Image, resample and rotation-encoding vocabularies (typing views over native sets).

``IMAGE_LAYOUTS``, ``RESAMPLES``, ``CROP_MODES``, ``CHANNEL_ORDERS`` and
``ROTATION_DIMS`` are each defined once, in the ``rlmesh-adapters`` crate
(``ImageLayout::ALL`` in ``v1/spec/layouts.rs``, ``RESAMPLES`` in
``v1/apply/image.rs``, ``CROP_MODES``/``CHANNEL_ORDERS`` in
``v1/spec/model/image.rs``, and ``RotationEncoding::dims`` in
``v1/spec/rotations.rs``); this module re-exports them through the native
bindings. ``ImageLayout``/``Resample``/``CropMode``/``ChannelOrder``/
``RotationEncoding`` are the Python-side typing views of the same value sets.
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
# ``gravity_xyz`` is a sink: the gravity direction in the body frame
# (``R_world_from_base^T · (0, 0, -1)``), which any rotation encoding converts
# into and nothing converts out of. Observation-side only; an actuator or a
# ``post_rotate`` naming it is a codec error.
RotationEncoding: TypeAlias = Literal[
    "quat_xyzw",
    "quat_wxyz",
    "axis_angle",
    "rot6d",
    "rot6d_rowmajor",
    "euler_xyz",
    "gravity_xyz",
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
# Where a state leaf's numbers come from (PROVENANCES in v1/spec/frames.rs): a
# physical sensor or its simulated equivalent, a state estimator, or simulator
# truth with no hardware counterpart. A disagreement is a resolve error.
Provenance: TypeAlias = Literal["sensed", "estimated", "privileged"]
# How a crop box is taken: ``"zoom"`` resamples the fractional box straight to
# the target (PIL's ``Image.resize(size, box=...)``), ``"slice"`` cuts an
# integer center box out first and resizes that.
CropMode: TypeAlias = Literal["zoom", "slice"]
# ``"bgr"`` swaps red and blue after the spatial ops (a 3-channel image only).
ChannelOrder: TypeAlias = Literal["rgb", "bgr"]

# A frame-history window: non-positive offsets from the current step, oldest
# first and ending at ``0`` ((-6, -4, -2, 0) is "every second frame of the last
# seven"). ``stride=`` is construction sugar for an evenly spaced one.
StackSpec: TypeAlias = tuple[int, ...]
# What fills a frame window before an episode has produced enough steps:
# ``"first"`` replicates the first frame, ``"black"`` is a raw 8-bit 0 frame
# pushed through the plan (so under ``normalize=(-1, 1)`` it lands on -1.0).
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
    "Provenance",
    "Reference",
    "Resample",
    "RotationEncoding",
    "StackPad",
    "StackSpec",
]
