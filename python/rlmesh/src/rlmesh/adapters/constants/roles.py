"""Semantic role vocabulary for matching env features to model inputs.

Roles are an open vocabulary: any string can be used as long as the env and
model specs agree. The two domains below are the registry of well-known
conventions that ship with RLMesh:

- Domain-agnostic roles (cameras, instruction text, joints).
- Arm manipulation roles (end-effector, gripper, and the bimanual ``_2``
  convention).

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

Bimanual convention: the first (or only) arm uses the unsuffixed roles; the
second arm uses the ``_2`` variants. Single-arm envs simply never declare
``_2`` roles, so model components targeting them resolve to zero fill
(observations) or dropped output dims (actions). By convention
``eef_pos``/``delta_eef_pos`` are 3-D Cartesian; gripper widths vary by
embodiment.

Values are defined once, in the ``rlmesh-adapters`` crate (``v1/roles/``);
this module re-exports them through the native bindings.
"""

from ..._rlmesh import (
    ACTION_DELTA_POS,
    ACTION_DELTA_POS_2,
    ACTION_DELTA_ROT,
    ACTION_DELTA_ROT_2,
    ACTION_GRIPPER,
    ACTION_GRIPPER_2,
    ACTION_JOINT_POS,
    ACTION_JOINT_VEL,
    EEF_POS,
    EEF_POS_2,
    EEF_ROT,
    EEF_ROT_2,
    GRIPPER_POS,
    GRIPPER_POS_2,
    IMAGE_PRIMARY,
    IMAGE_SECONDARY,
    IMAGE_WRIST,
    INSTRUCTION,
    JOINT_POS,
    JOINT_VEL,
)

__all__ = [
    "ACTION_DELTA_POS",
    "ACTION_DELTA_POS_2",
    "ACTION_DELTA_ROT",
    "ACTION_DELTA_ROT_2",
    "ACTION_GRIPPER",
    "ACTION_GRIPPER_2",
    "ACTION_JOINT_POS",
    "ACTION_JOINT_VEL",
    "EEF_POS",
    "EEF_POS_2",
    "EEF_ROT",
    "EEF_ROT_2",
    "GRIPPER_POS",
    "GRIPPER_POS_2",
    "IMAGE_PRIMARY",
    "IMAGE_SECONDARY",
    "IMAGE_WRIST",
    "INSTRUCTION",
    "JOINT_POS",
    "JOINT_VEL",
]
