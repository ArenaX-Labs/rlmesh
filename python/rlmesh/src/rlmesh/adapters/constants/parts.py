"""Suggested spellings for ``part``: the place on the body a role repeats at.

A ``part`` is an identity key on a leaf (``StateTag``, ``ImageTag``, ``Field``,
``Actuator``, ``Image``, ``State``): any identifier both sides agree on, matched
verbatim. Two arms are ``EEF_POS`` under ``part=LEFT_ARM`` and under
``part=RIGHT_ARM``. The constants below are conventions, not a registry: a part
outside them resolves the same way and draws no advisory.

Values are defined once, in the ``rlmesh-adapters`` crate (``roles/parts.rs``);
this module re-exports them through the native bindings.
"""

from ..._rlmesh import (
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
    "BASE",
    "HEAD",
    "LEFT_ARM",
    "LEFT_LEG",
    "PARTS",
    "RIGHT_ARM",
    "RIGHT_LEG",
    "TORSO",
]
