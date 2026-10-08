"""Semantic role vocabulary for matching env features to model inputs.

A role is ``<kind>/<name>``, and the kinds are closed: ``image/``,
``proprio/``, ``text/``, ``action/``, and ``command/``. Any other prefix is
refused when a spec is authored, joined, or resolved. A role this registry
does not list still resolves, but only when the env and model agree on its
exact string, so the join nudges it toward a registered name.

A deliberately non-standard role goes under the ``x/`` escape (for example
``x/target_pos``): it is never nudged and passes the strict publish gate. A
role with no ``/`` names no kind at all; it is still accepted, but it skips
the kind check and draws an advisory recommending a kind prefix or ``x/``.

There is no registered goal role yet: by convention, a goal or target
position is ``x/target_pos``.

The registry ships three domains:

- Domain-agnostic roles (cameras, instruction text, joints).
- Arm manipulation roles (end-effector pose, gripper, the six-axis wrench).
- Body roles (the floating base: gyro, orientation, and the velocity command
  under the ``command/`` kind). Projected gravity is not a role but the
  ``gravity_xyz`` encoding of ``BASE_ROT``; the base linear velocity is
  deliberately unregistered (a real robot only has an estimate of it).

Registry policy: a domain earns its roles here when its first real env/model
pair lands; until then its specs use ``x/`` roles. Role strings are wire
format -- they are matched verbatim between independently authored specs and
must never be renamed once released. The prefix names a feature kind, not a
domain: domains sharing a role (e.g. ``proprio/joint_pos`` in both
manipulation and locomotion) is intentional.

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

Values are defined once, in the ``rlmesh-adapters`` crate (``src/roles/``);
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
