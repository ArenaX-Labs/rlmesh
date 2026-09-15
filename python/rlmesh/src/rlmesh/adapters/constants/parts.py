"""The parts vocabulary: the physical places a role can repeat at.

A ``part`` is an identity key on a leaf (``StateTag``, ``ImageTag``, ``Field``,
``Actuator``, ``Image``, ``State``), not a per-robot name: ``LEFT_ARM`` is
where a second ``EEF_POS`` lives, never "the Franka". Like roles, parts are an
open vocabulary matched verbatim; the constants below are the registered ones,
grown the way roles are (when a real env and model pair needs the slot). Any
other part draws an authoring nudge and the managed strict tier rejects it,
unless written under the ``x/`` escape prefix. ``ARM_2`` is the part the legacy
``_2`` roles spell (``EEF_POS_2`` is ``EEF_POS`` under ``part=ARM_2``).

Values are defined once, in the ``rlmesh-adapters`` crate (``roles/parts.rs``);
this module re-exports them through the native bindings.
"""

from ..._rlmesh import (
    ARM_2,
    BASE,
    HEAD,
    LEFT_ARM,
    LEFT_LEG,
    PARTS,
    RIGHT_ARM,
    RIGHT_LEG,
    TORSO,
)

__all__ = [
    "ARM_2",
    "BASE",
    "HEAD",
    "LEFT_ARM",
    "LEFT_LEG",
    "PARTS",
    "RIGHT_ARM",
    "RIGHT_LEG",
    "TORSO",
]
