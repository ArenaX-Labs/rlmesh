"""Semantic role vocabulary for matching env features to model inputs.

Roles are an open vocabulary: any string can be used as long as the env and
model specs agree. The two domains below are the registry of well-known
conventions that ship with RLMesh:

- Domain-agnostic roles (cameras, instruction text, joints).
- Arm manipulation roles (end-effector pose, gripper, the six-axis wrench).
- Body roles (the floating base: gyro, orientation, and the velocity command
  under the ``command/`` kind). Projected gravity is not a role but the
  ``gravity_xyz`` encoding of ``BASE_ROT``; the base linear velocity is
  deliberately unregistered (a real robot only has an estimate of it).

Registry policy: a domain earns its roles here when its first real env/model
pair lands; until then its specs use ad-hoc strings. Role strings are wire
format -- they are matched verbatim between independently authored specs and
must never be renamed once released. Strings carry feature-kind prefixes
(``proprio/``, ``action/``, ``image/``, ``text/``), not domain prefixes:
domains sharing a role (e.g. ``proprio/joint_pos`` in both manipulation and
locomotion) is intentional.

Width conventions: the author always pins ``dim`` explicitly; a registered role
with a fixed canonical width (e.g. ``eef_pos``/``delta_eef_pos`` are 3-D
Cartesian) now *validates* that declared dim and rejects a mismatch, but never
supplies it. Rotation widths follow the declared encoding (see
``ROTATION_DIMS``); other widths vary by embodiment, which is what
``dim``/``index`` selection on components is for.

Multiple arms: a role repeats under a ``part`` (see :mod:`.parts`), so a second
arm is ``EEF_POS`` under ``part=RIGHT_ARM``. When an env has two leaves of one
role, a model must name the part it wants. By convention
``eef_pos``/``delta_eef_pos`` are 3-D Cartesian; gripper widths vary by
embodiment.

Values are defined once, in the ``rlmesh-adapters`` crate (``v1/roles/``);
this module re-exports them through the native bindings.
"""

from ..._rlmesh import (
    ACTION_DELTA_POS,
    ACTION_DELTA_ROT,
    ACTION_EEF_POS,
    ACTION_EEF_ROT,
    ACTION_GRIPPER,
    ACTION_JOINT_POS,
    ACTION_JOINT_VEL,
    BASE_ANG_VEL,
    BASE_ROT,
    COMMAND_BASE_VEL,
    EEF_POS,
    EEF_ROT,
    EEF_WRENCH,
    GRIPPER_POS,
    IMAGE_PRIMARY,
    IMAGE_SECONDARY,
    IMAGE_WRIST,
    INSTRUCTION,
    JOINT_POS,
    JOINT_VEL,
)

__all__ = [
    "ACTION_DELTA_POS",
    "ACTION_DELTA_ROT",
    "ACTION_EEF_POS",
    "ACTION_EEF_ROT",
    "ACTION_GRIPPER",
    "ACTION_JOINT_POS",
    "ACTION_JOINT_VEL",
    "BASE_ANG_VEL",
    "BASE_ROT",
    "COMMAND_BASE_VEL",
    "EEF_POS",
    "EEF_ROT",
    "EEF_WRENCH",
    "GRIPPER_POS",
    "IMAGE_PRIMARY",
    "IMAGE_SECONDARY",
    "IMAGE_WRIST",
    "INSTRUCTION",
    "JOINT_POS",
    "JOINT_VEL",
]
