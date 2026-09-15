"""Embodiment profiles: the shipped joint-label tuples ``labels=`` is written from.

A profile is data, not a wire type and not a resolver input. An environment
writes ``labels=embodiments.GO2.joints``, a model writes the same tuple or a
subset (in its own training order), and the wire still carries plain strings.
The resolver never consults a profile: two sides agree on label strings or
they do not. What profiles are for is authoring (nobody should retype twelve
joint names) and the label lint: a tuple that matches no shipped profile as a
set draws the same nudge an ad-hoc role does, and the managed
``--require-labels`` tier refuses it.

Label spelling is ``<part>_<joint>`` snake case, matching the Unitree SDK and
Isaac Lab joint names. The rows are defined once, in the ``rlmesh-adapters``
crate (``roles/embodiments.rs``); this module builds its constants from them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._rlmesh import EMBODIMENTS


@dataclass(frozen=True)
class EmbodimentProfile:
    """A named embodiment: its registered parts and its joint labels in order.

    Attributes:
        name: The profile's stable name, e.g. ``"unitree_go2"``.
        parts: The registered body parts this embodiment has.
        joints: Its joint labels in the vendor's canonical (SDK motor) order.
    """

    name: str
    parts: tuple[str, ...]
    joints: tuple[str, ...]


PROFILES: tuple[EmbodimentProfile, ...] = tuple(
    EmbodimentProfile(name=name, parts=tuple(parts), joints=tuple(joints))
    for name, parts, joints in EMBODIMENTS
)
_BY_NAME = {profile.name: profile for profile in PROFILES}

#: Unitree Go2: 12 joints in SDK motor order (FR, FL, RR, RL x hip, thigh, calf).
GO2: EmbodimentProfile = _BY_NAME["unitree_go2"]
#: Unitree G1, 29-DoF: two 6-DoF legs, a 3-DoF waist, two 7-DoF arms.
G1_29DOF: EmbodimentProfile = _BY_NAME["unitree_g1_29dof"]
#: Franka Emika Panda: ``panda_joint1`` .. ``panda_joint7``.
FRANKA_PANDA: EmbodimentProfile = _BY_NAME["franka_panda"]

__all__ = ["FRANKA_PANDA", "G1_29DOF", "GO2", "PROFILES", "EmbodimentProfile"]
