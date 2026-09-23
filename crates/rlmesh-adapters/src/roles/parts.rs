//! Suggested part names: where a role repeats on a body.
//!
//! A `part` is any identifier both sides of a join agree on exactly; there is
//! no registry and nothing here validates one. These constants are
//! conventions for the common places a role repeats, nothing more.

pub const LEFT_ARM: &str = "left_arm";
pub const RIGHT_ARM: &str = "right_arm";
pub const HEAD: &str = "head";
pub const TORSO: &str = "torso";
pub const BASE: &str = "base";
pub const LEFT_LEG: &str = "left_leg";
pub const RIGHT_LEG: &str = "right_leg";

/// The suggested part names, for consumers exporting the conventions.
pub const PARTS: [&str; 7] = [LEFT_ARM, RIGHT_ARM, HEAD, TORSO, BASE, LEFT_LEG, RIGHT_LEG];
