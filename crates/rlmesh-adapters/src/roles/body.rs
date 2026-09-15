//! Body roles: the floating base of a legged or wheeled robot.
//!
//! Earned by the Unitree Go2 env and the rl_sar `robot_lab` locomotion
//! checkpoint. Each is a raw sensed or commanded quantity: the gyro reading,
//! the IMU orientation, and the velocity setpoint the runner hands the policy.
//! Projected gravity is *not* a role -- it is the `gravity_xyz` encoding of
//! [`BASE_ROT`] (see `spec::rotations`), and the base linear velocity is
//! deliberately unregistered: a real robot has only an estimate of it, so a
//! role would silently mean two different things in sim and on hardware.

use super::registry::{DimLaw, RoleDef};
use crate::spec::{FrameLaw, ReferenceLaw};

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
