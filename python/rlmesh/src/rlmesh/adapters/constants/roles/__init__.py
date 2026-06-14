"""Semantic role vocabulary for matching env features to model inputs.

Roles are an open vocabulary: any string can be used as long as the env and
model specs agree. The modules below are the registry of well-known
conventions that ship with RLMesh, organized by domain:

- :mod:`.core` -- domain-agnostic roles (cameras, instruction text, joints).
- :mod:`.manipulation` -- arm manipulation roles (end-effector, gripper,
  and the bimanual ``_2`` convention).

Registry policy: a domain earns a module here when its first real env/model
pair lands; until then its specs use ad-hoc strings. Role strings are wire
format -- they are matched verbatim between independently authored specs
and must never be renamed once released. Strings carry feature-kind
prefixes (``proprio/``, ``action/``, ``image/``, ``text/``), not domain
prefixes: domains sharing a role (e.g. ``proprio/joint_pos`` in both
manipulation and locomotion) is intentional.

Width conventions: roles do not imply dims mechanically -- specs pin widths
explicitly where they matter. Rotation widths follow the declared encoding
(see ``ROTATION_DIMS``); other widths vary by embodiment, which is what
``dim``/``index`` selection on components is for.
"""

from .core import (
    IMAGE_PRIMARY,
    IMAGE_SECONDARY,
    INSTRUCTION,
    JOINT_POS,
    JOINT_VEL,
)
from .manipulation import (
    ACTION_DELTA_POS,
    ACTION_DELTA_POS_2,
    ACTION_DELTA_ROT,
    ACTION_DELTA_ROT_2,
    ACTION_GRIPPER,
    ACTION_GRIPPER_2,
    EEF_POS,
    EEF_POS_2,
    EEF_ROT,
    EEF_ROT_2,
    GRIPPER_POS,
    GRIPPER_POS_2,
    IMAGE_WRIST,
)

__all__ = [
    "ACTION_DELTA_POS",
    "ACTION_DELTA_POS_2",
    "ACTION_DELTA_ROT",
    "ACTION_DELTA_ROT_2",
    "ACTION_GRIPPER",
    "ACTION_GRIPPER_2",
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
