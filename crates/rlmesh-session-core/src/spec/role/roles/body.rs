//! Body roles: the floating base of a legged or wheeled robot.
//!
//! Each is a raw sensed or commanded quantity. Projected gravity is not a role: it is the
//! `gravity_xyz` encoding of [`BASE_ROT`]. Base linear velocity is deliberately unregistered, since
//! a real robot has only an estimate of it.

#[allow(unused_imports)]
use crate::prelude::*;

use super::registry::{DimLaw, RoleDef};
use crate::spec::role::frames::{FrameLaw, ReferenceLaw};

/// Base angular velocity (the IMU gyro), 3-D, expressed in a frame.
pub const BASE_ANG_VEL: &str = "proprio/base_ang_vel";
/// Base orientation, width by its rotation encoding, expressed in a frame.
pub const BASE_ROT: &str = "proprio/base_rot";
/// The velocity command the policy tracks: `[vx, vy, wz]` in a frame. The
/// numeric sibling of `text/instruction`, under the `command/` kind.
pub const COMMAND_BASE_VEL: &str = "command/base_vel";

/// Body role table. The base velocities are `Fixed(3)`; the orientation
/// defers to its encoding (`ByEncoding`). All three are framed.
pub const ROLES: &[RoleDef] = &[
    RoleDef {
        name: BASE_ANG_VEL,
        dim: DimLaw::Fixed(3),
        doc: "base angular velocity (gyro)",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: BASE_ROT,
        dim: DimLaw::ByEncoding,
        doc: "base orientation",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
    RoleDef {
        name: COMMAND_BASE_VEL,
        dim: DimLaw::Fixed(3),
        doc: "commanded base velocity [vx, vy, wz]",
        frame: FrameLaw::Framed,
        reference: ReferenceLaw::Unreferenced,
    },
];
