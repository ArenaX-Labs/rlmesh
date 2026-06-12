"""Rotation encoding vocabulary and per-encoding dimensions.

``ROTATION_DIMS`` is defined once, in the ``rlmesh-adapters`` crate
(``RotationEncoding::dims`` in ``v1/spec/rotations.rs``); this module
re-exports it through the native bindings. ``RotationEncoding`` is the
Python-side typing view of the same value set.
"""

from typing import Literal, TypeAlias

from ..._rlmesh import ROTATION_DIMS

RotationEncoding: TypeAlias = Literal["quat_xyzw", "quat_wxyz", "axis_angle", "rot6d"]

__all__ = ["ROTATION_DIMS", "RotationEncoding"]
