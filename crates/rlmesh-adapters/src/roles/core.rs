//! Domain-agnostic roles shared across environment/model domains.

use super::registry::{DimLaw, RoleDef};

pub const IMAGE_PRIMARY: &str = "image/primary";
pub const IMAGE_SECONDARY: &str = "image/secondary";

pub const INSTRUCTION: &str = "text/instruction";

pub const JOINT_POS: &str = "proprio/joint_pos";
pub const JOINT_VEL: &str = "proprio/joint_vel";

// Joint-space action (e.g. gym-aloha's absolute joint positions). Blessed when
// its first real env/model pair landed (joint-control catalog envs).
pub const ACTION_JOINT_POS: &str = "action/joint_pos";
pub const ACTION_JOINT_VEL: &str = "action/joint_vel";

/// Core domain role table. Joint widths vary by embodiment (DoF), so joint roles
/// are `Variable`; images and text carry no numeric dim law.
pub const ROLES: &[RoleDef] = &[
    RoleDef {
        name: IMAGE_PRIMARY,
        dim: DimLaw::Variable,
        doc: "primary camera frame",
    },
    RoleDef {
        name: IMAGE_SECONDARY,
        dim: DimLaw::Variable,
        doc: "secondary camera frame",
    },
    RoleDef {
        name: INSTRUCTION,
        dim: DimLaw::Variable,
        doc: "natural-language task instruction",
    },
    RoleDef {
        name: JOINT_POS,
        dim: DimLaw::Variable,
        doc: "joint positions (DoF varies by embodiment)",
    },
    RoleDef {
        name: JOINT_VEL,
        dim: DimLaw::Variable,
        doc: "joint velocities (DoF varies by embodiment)",
    },
    RoleDef {
        name: ACTION_JOINT_POS,
        dim: DimLaw::Variable,
        doc: "commanded joint positions (DoF varies by embodiment)",
    },
    RoleDef {
        name: ACTION_JOINT_VEL,
        dim: DimLaw::Variable,
        doc: "commanded joint velocities (DoF varies by embodiment)",
    },
];
