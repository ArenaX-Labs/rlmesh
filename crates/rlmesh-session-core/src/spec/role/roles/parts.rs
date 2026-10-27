//! Suggested part names: where a role repeats on a body.
//!
//! A `part` is any identifier both sides of a join agree on exactly; there is
//! no registry and nothing here validates one. These constants are
//! conventions for the common places a role repeats, nothing more.

#[allow(unused_imports)]
use crate::prelude::*;

/// The left arm.
pub const LEFT_ARM: &str = "left_arm";
/// The right arm.
pub const RIGHT_ARM: &str = "right_arm";
/// The head.
pub const HEAD: &str = "head";
/// The torso.
pub const TORSO: &str = "torso";
/// The mobile base.
pub const BASE: &str = "base";
/// The left leg.
pub const LEFT_LEG: &str = "left_leg";
/// The right leg.
pub const RIGHT_LEG: &str = "right_leg";

/// The suggested part names, for consumers exporting the conventions.
pub const PARTS: [&str; 7] = [LEFT_ARM, RIGHT_ARM, HEAD, TORSO, BASE, LEFT_LEG, RIGHT_LEG];
