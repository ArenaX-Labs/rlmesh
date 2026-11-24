"""Shared adapter constants: semantic roles, body parts and metadata keys.

Metadata keys are versioned: within ``v1`` the JSON spec format evolves
additively only, and a breaking change ships under a new key. Readers dispatch
on the key alone. Keys and roles are defined in the ``rlmesh-adapters`` crate
and re-exported through the native bindings.
"""

from ..._rlmesh import (
    ENV_BRANCH_METADATA_KEY,
    ENV_METADATA_KEY,
    MODEL_METADATA_KEY,
)
from .parts import (
    BASE,
    HEAD,
    LEFT_ARM,
    LEFT_LEG,
    PARTS,
    RIGHT_ARM,
    RIGHT_LEG,
    TORSO,
)
from .roles import (
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
    "BASE",
    "BASE_ANG_VEL",
    "BASE_ROT",
    "COMMAND_BASE_VEL",
    "EEF_POS",
    "EEF_ROT",
    "EEF_WRENCH",
    "ENV_BRANCH_METADATA_KEY",
    "ENV_METADATA_KEY",
    "GRIPPER_POS",
    "HEAD",
    "IMAGE_PRIMARY",
    "IMAGE_SECONDARY",
    "IMAGE_WRIST",
    "INSTRUCTION",
    "JOINT_POS",
    "JOINT_VEL",
    "LEFT_ARM",
    "LEFT_LEG",
    "MODEL_METADATA_KEY",
    "PARTS",
    "RIGHT_ARM",
    "RIGHT_LEG",
    "TORSO",
]
