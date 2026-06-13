//! Arm manipulation roles (single-arm and bimanual).
//!
//! Bimanual convention: the first (or only) arm uses the unsuffixed
//! roles; the second arm uses the `_2` variants. Single-arm envs simply
//! never declare `_2` roles, so model components targeting them resolve
//! to zero fill (observations) or dropped output dims (actions).
//!
//! By convention `eef_pos`/`delta_eef_pos` are 3-D Cartesian; gripper
//! widths vary by embodiment.

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
