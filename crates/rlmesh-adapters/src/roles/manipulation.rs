//! Arm manipulation roles (single-arm and bimanual).
//!
//! Bimanual convention: each arm's leaf carries the base role and a `part`
//! (`left_arm`, `right_arm`; see [`parts`](super::parts)). Single-arm envs
//! simply never declare a second arm.
//!
//! By convention `eef_pos`/`delta_eef_pos` are 3-D Cartesian; gripper
//! widths vary by embodiment. `eef_wrench` is the six-axis force/torque at
//! the end effector, force first (`[fx, fy, fz, tx, ty, tz]`), framed: a
//! wrist-mounted sensor reads it in `tool`, a controller's estimate is usually
//! in `robot_base`, and the two are different numbers for the same contact.

use super::registry::{DimLaw, RoleDef};
use crate::spec::{FrameLaw, ReferenceLaw};

pub const IMAGE_WRIST: &str = "image/wrist";

pub const EEF_POS: &str = "proprio/eef_pos";
pub const EEF_ROT: &str = "proprio/eef_rot";
pub const GRIPPER_POS: &str = "proprio/gripper";
/// End-effector wrench `[fx, fy, fz, tx, ty, tz]`, expressed in a frame. The
/// raw force/torque reading (a wrist sensor, or a simulated one), never a
/// contact model's output.
pub const EEF_WRENCH: &str = "proprio/eef_wrench";

pub const ACTION_DELTA_POS: &str = "action/delta_eef_pos";
pub const ACTION_DELTA_ROT: &str = "action/delta_eef_rot";
pub const ACTION_GRIPPER: &str = "action/gripper";

/// Absolute end-effector targets, for policies trained on controller goals
/// rather than per-step deltas (X-VLA, RoboTwin's `ee` action type). The env
/// runs its Cartesian controller in absolute mode and receives the target
/// verbatim; the frame is the env's `proprio/eef_*` frame.
pub const ACTION_EEF_POS: &str = "action/eef_pos";
pub const ACTION_EEF_ROT: &str = "action/eef_rot";

/// Manipulation role table. `eef_pos`/`delta_eef_pos` are 3-D Cartesian
/// (`Fixed(3)`); rotations defer to their encoding (`ByEncoding`); gripper widths
/// vary by embodiment (`Variable`).
pub const ROLES: &[RoleDef] = &[
    RoleDef {
        name: IMAGE_WRIST,
        dim: DimLaw::Variable,
        doc: "wrist camera frame",
        frame: FrameLaw::Frameless,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: EEF_POS,
        dim: DimLaw::Fixed(3),
        doc: "end-effector Cartesian position",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: EEF_ROT,
        dim: DimLaw::ByEncoding,
        doc: "end-effector rotation",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: GRIPPER_POS,
        dim: DimLaw::Variable,
        doc: "gripper width (finger count varies)",
        frame: FrameLaw::Frameless,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: EEF_WRENCH,
        dim: DimLaw::Fixed(6),
        doc: "end-effector wrench [fx, fy, fz, tx, ty, tz]",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: ACTION_DELTA_POS,
        dim: DimLaw::Fixed(3),
        doc: "Cartesian end-effector position delta",
        frame: FrameLaw::Frameless,
        reference: ReferenceLaw::Referenced,
    },
    RoleDef {
        name: ACTION_DELTA_ROT,
        dim: DimLaw::ByEncoding,
        doc: "end-effector rotation delta",
        frame: FrameLaw::Frameless,
        reference: ReferenceLaw::Referenced,
    },
    RoleDef {
        name: ACTION_GRIPPER,
        dim: DimLaw::Variable,
        doc: "gripper command",
        frame: FrameLaw::Frameless,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: ACTION_EEF_POS,
        dim: DimLaw::Fixed(3),
        doc: "absolute Cartesian end-effector position target",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: ACTION_EEF_ROT,
        dim: DimLaw::ByEncoding,
        doc: "absolute end-effector rotation target",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
];
