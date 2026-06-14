"""Arm manipulation roles (single-arm and bimanual).

Bimanual convention: the first (or only) arm uses the unsuffixed roles; the
second arm uses the ``_2`` variants. Single-arm envs simply never declare
``_2`` roles, so model components targeting them resolve to zero fill
(observations) or dropped output dims (actions).

By convention ``eef_pos``/``delta_eef_pos`` are 3-D Cartesian; rotation
widths follow the declared encoding (see ``ROTATION_DIMS``); gripper widths
vary by embodiment, which is what ``dim``/``index`` selection on components
is for.

Values are defined once, in the ``rlmesh-adapters`` crate
(``v1/roles/manipulation.rs``); this module re-exports them through the
native bindings.
"""

from ...._rlmesh import (
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
    "IMAGE_WRIST",
]
