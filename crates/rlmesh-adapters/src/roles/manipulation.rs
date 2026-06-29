//! Arm manipulation roles (single-arm and bimanual).
//!
//! Bimanual convention: the first (or only) arm uses the unsuffixed
//! roles; the second arm uses the `_2` variants. Single-arm envs simply
//! never declare `_2` roles, so model components targeting them resolve
//! to zero fill (observations) or dropped output dims (actions).
//!
//! By convention `eef_pos`/`delta_eef_pos` are 3-D Cartesian; gripper
//! widths vary by embodiment.

use super::registry::{DimLaw, RoleDef};

pub const IMAGE_WRIST: &str = "image/wrist";

pub const EEF_POS: &str = "proprio/eef_pos";
pub const EEF_ROT: &str = "proprio/eef_rot";
pub const GRIPPER_POS: &str = "proprio/gripper";

pub const EEF_POS_2: &str = "proprio/eef_pos_2";
pub const EEF_ROT_2: &str = "proprio/eef_rot_2";
pub const GRIPPER_POS_2: &str = "proprio/gripper_2";

pub const ACTION_DELTA_POS: &str = "action/delta_eef_pos";
pub const ACTION_DELTA_ROT: &str = "action/delta_eef_rot";
pub const ACTION_GRIPPER: &str = "action/gripper";

pub const ACTION_DELTA_POS_2: &str = "action/delta_eef_pos_2";
pub const ACTION_DELTA_ROT_2: &str = "action/delta_eef_rot_2";
pub const ACTION_GRIPPER_2: &str = "action/gripper_2";

/// Manipulation role table. `eef_pos`/`delta_eef_pos` are 3-D Cartesian
/// (`Fixed(3)`); rotations defer to their encoding (`ByEncoding`); gripper widths
/// vary by embodiment (`Variable`). The `_2` series mirrors the first arm.
pub const ROLES: &[RoleDef] = &[
    RoleDef {
        name: IMAGE_WRIST,
        dim: DimLaw::Variable,
        doc: "wrist camera frame",
    },
    RoleDef {
        name: EEF_POS,
        dim: DimLaw::Fixed(3),
        doc: "end-effector Cartesian position",
    },
    RoleDef {
        name: EEF_ROT,
        dim: DimLaw::ByEncoding,
        doc: "end-effector rotation",
    },
    RoleDef {
        name: GRIPPER_POS,
        dim: DimLaw::Variable,
        doc: "gripper width (finger count varies)",
    },
    RoleDef {
        name: EEF_POS_2,
        dim: DimLaw::Fixed(3),
        doc: "second-arm end-effector position",
    },
    RoleDef {
        name: EEF_ROT_2,
        dim: DimLaw::ByEncoding,
        doc: "second-arm end-effector rotation",
    },
    RoleDef {
        name: GRIPPER_POS_2,
        dim: DimLaw::Variable,
        doc: "second-arm gripper width",
    },
    RoleDef {
        name: ACTION_DELTA_POS,
        dim: DimLaw::Fixed(3),
        doc: "Cartesian end-effector position delta",
    },
    RoleDef {
        name: ACTION_DELTA_ROT,
        dim: DimLaw::ByEncoding,
        doc: "end-effector rotation delta",
    },
    RoleDef {
        name: ACTION_GRIPPER,
        dim: DimLaw::Variable,
        doc: "gripper command",
    },
    RoleDef {
        name: ACTION_DELTA_POS_2,
        dim: DimLaw::Fixed(3),
        doc: "second-arm Cartesian position delta",
    },
    RoleDef {
        name: ACTION_DELTA_ROT_2,
        dim: DimLaw::ByEncoding,
        doc: "second-arm rotation delta",
    },
    RoleDef {
        name: ACTION_GRIPPER_2,
        dim: DimLaw::Variable,
        doc: "second-arm gripper command",
    },
];
