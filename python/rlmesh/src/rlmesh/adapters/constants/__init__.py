"""Shared adapter constants: semantic roles and metadata keys.

Metadata keys are versioned like protobuf packages: within ``v1`` the JSON spec
format evolves additively only (new optional fields with defaults), and a
breaking format change ships under a new ``v2`` key. Publishers may carry
multiple versions in one metadata mapping during a migration; readers dispatch
on the key alone, without parsing payloads. Both the keys and the role
vocabulary are defined once in the ``rlmesh-adapters`` crate and re-exported here
through the native bindings.
"""

from ..._rlmesh import ENV_METADATA_KEY, MODEL_METADATA_KEY
from .roles import (
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
    "EEF_POS",
    "EEF_POS_2",
    "EEF_ROT",
    "EEF_ROT_2",
    "ENV_METADATA_KEY",
    "GRIPPER_POS",
    "GRIPPER_POS_2",
    "IMAGE_PRIMARY",
    "IMAGE_SECONDARY",
    "IMAGE_WRIST",
    "INSTRUCTION",
    "JOINT_POS",
    "JOINT_VEL",
    "MODEL_METADATA_KEY",
]
