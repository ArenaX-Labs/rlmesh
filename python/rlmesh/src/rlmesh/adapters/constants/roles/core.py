"""Domain-agnostic roles shared across environment/model domains.

Values are defined once, in the ``rlmesh-adapters`` crate
(``v1/roles/core.rs``); this module re-exports them through the native
bindings.
"""

from ...._rlmesh import (
    IMAGE_PRIMARY,
    IMAGE_SECONDARY,
    INSTRUCTION,
    JOINT_POS,
    JOINT_VEL,
)

__all__ = [
    "IMAGE_PRIMARY",
    "IMAGE_SECONDARY",
    "INSTRUCTION",
    "JOINT_POS",
    "JOINT_VEL",
]
