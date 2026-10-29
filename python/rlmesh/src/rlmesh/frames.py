"""Cross-robot geometry in a scene (scene design §5).

Each place's pose is data: the static pose of one of its robot's frames in
the scene frame ``world`` (z up). RLMesh never transforms a value between
frames; a policy that needs another robot's frame composes two places'
poses with :func:`between`, as the eyeball rig maps a camera ray into an
eye's frame::

    t = rlmesh.frames.between(places["scene"], places["eyes/left"])
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ._rlmesh import frames_between

__all__ = ["between"]


def _pose(value: Mapping[str, Any]) -> Mapping[str, Any]:
    pose: object = value.get("pose", value)
    if not isinstance(pose, Mapping):
        raise TypeError("a place's pose is a mapping with xyz and quat_xyzw")
    pose = cast(Mapping[str, Any], pose)
    if "xyz" not in pose or "quat_xyzw" not in pose:
        raise TypeError(
            "a pose is a mapping with xyz and quat_xyzw, or a place with such a pose"
        )
    return pose


def between(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, list[float]]:
    """The static rigid transform from place ``a``'s frame to place ``b``'s.

    ``a`` and ``b`` are places (with a ``pose``) or poses: ``xyz`` in metres
    and a unit ``quat_xyzw``, both in ``world``. The result,
    ``T_b←a = T_world←b⁻¹ · T_world←a`` in 64 bits, is a pose of the same
    shape: a point ``p`` in ``a``'s frame is ``R(quat_xyzw) · p + xyz`` in
    ``b``'s, and the quaternion has ``w ≥ 0``.
    """
    pa, pb = _pose(a), _pose(b)
    xyz, quat = frames_between(pa["xyz"], pa["quat_xyzw"], pb["xyz"], pb["quat_xyzw"])
    return {"xyz": list(xyz), "quat_xyzw": list(quat)}
