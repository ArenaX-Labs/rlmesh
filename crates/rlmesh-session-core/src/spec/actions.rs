//! A policy's actions, built by kind: one constructor per commandable kind,
//! a setter per key.
//!
//! Each builder stands for one action entry of the policy map, with only
//! the keys set; [`Policy::builder`](crate::spec::Policy::builder) takes it
//! beside entries written as JSON. The address is set with `at`, as for
//! [`inputs`](crate::spec::inputs); without one, the entry binds any part
//! of its kind. Values the types cannot rule out (a chunk of 0, a rate
//! above 10 kHz) are refused when the policy is built, naming the entry.
//!
//! ```
//! use core::time::Duration;
//!
//! use rlmesh_session_core::spec::actions::{self, ChunkStart, JointCommand};
//! use rlmesh_session_core::spec::Policy;
//!
//! let policy = Policy::builder()
//!     .hz(30)
//!     .input("prompt", serde_json::json!("text"))
//!     .action(
//!         "left",
//!         actions::joints()
//!             .at("gimbal-left:servos")
//!             .command(JointCommand::Position)
//!             .hz(50)
//!             .chunk(4)
//!             .t0(ChunkStart::Send)
//!             .ttl(Duration::from_millis(150)),
//!     )
//!     .action("grip", actions::gripper().at("gimbal-left:grip").open(0).closed(1))
//!     .build()?;
//! assert_eq!(policy.actions[0].hz, Some(50.0));
//! # Ok::<(), String>(())
//! ```
//!
//! Every action takes `command` (the part's command mode: the binding
//! refuses a part in another), `hz` (the rate of the actions in a chunk;
//! default the policy's), `chunk` (actions per chunk, default 1), `t0`
//! (when a chunk's first action is due), `ttl` (a cap on a chunk's life),
//! `optional` (may stay unbound, with `drop` allowed) and `allow`. Numeric
//! kinds take `unit`, `scale`, `offset` and `range` as inputs do.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::spec::policy::{Draft, builder, named};

pub use crate::spec::Rotation;
pub use crate::spec::policy::{Allow, Array, ChunkStart, PerElement};
pub use crate::wire::part::DType;

named! {
    /// A joint set's command mode.
    JointCommand {
        /// Position targets.
        Position = "position",
        /// Velocity targets.
        Velocity = "velocity",
        /// Torques.
        Torque = "torque",
    }
}

named! {
    /// A pose's command mode.
    PoseCommand {
        /// Targets in the entry's frame.
        Absolute = "absolute",
        /// Offsets from the [`Reference`].
        Delta = "delta",
    }
}

named! {
    /// What a pose delta is relative to.
    Reference {
        /// The measured pose.
        Current = "current",
        /// The last commanded target.
        Target = "target",
    }
}

named! {
    /// A wrench's command mode: its only one.
    WrenchCommand {
        /// A wrench to apply.
        Wrench = "wrench",
    }
}

named! {
    /// A gripper's command mode: its only one.
    GripperCommand {
        /// Opening targets.
        Position = "position",
    }
}

named! {
    /// A switch's or value's command mode: their only one.
    SetCommand {
        /// The value to set.
        Set = "set",
    }
}

named! {
    /// A base's command mode: its only one.
    VelocityCommand {
        /// Twist targets.
        Velocity = "velocity",
    }
}

builder! {
    /// A joint set action; see [`joints`].
    Joints => ActionEntry [
        command(JointCommand), hz, chunk, names, positional, count, unit, scale, offset, range,
        t0, ttl, optional, allow,
    ]
}

builder! {
    /// A hand action; see [`hand`].
    Hand => ActionEntry [
        command(JointCommand), hz, chunk, names, positional, count, unit, scale, offset, range,
        t0, ttl, optional, allow,
    ]
}

builder! {
    /// A wrench action; see [`wrench`].
    Wrench => ActionEntry [
        command(WrenchCommand), frame, hz, chunk, unit, scale, offset, range,
        t0, ttl, optional, allow,
    ]
}

builder! {
    /// A pose action; see [`pose`].
    Pose => ActionEntry [
        command(PoseCommand), reference, frame, child, rotation, hz, chunk, unit, scale, offset,
        t0, ttl, optional, allow,
    ]
}

builder! {
    /// A gripper action; see [`gripper`].
    Gripper => ActionEntry [
        command(GripperCommand), hz, chunk, open, closed, scale, offset, range,
        t0, ttl, optional, allow,
    ]
}

builder! {
    /// A switch action; see [`switch`].
    Switch => ActionEntry [command(SetCommand), hz, chunk, t0, ttl, optional, allow]
}

builder! {
    /// A value action; see [`value`].
    Value => ActionEntry [
        command(SetCommand), hz, chunk, unit, scale, offset, range, t0, ttl, optional, allow,
    ]
}

builder! {
    /// A base twist action; see [`velocity`].
    Velocity => ActionEntry [
        command(VelocityCommand), frame, hz, chunk, unit, scale, offset, range,
        t0, ttl, optional, allow,
    ]
}

builder! {
    /// A custom kind's action; see [`custom`].
    Custom => ActionEntry [
        field, custom_command, frame, hz, chunk, unit, scale, offset, range,
        t0, ttl, optional, allow,
    ]
}

/// Commands for a set of joints: `position`, `velocity` or `torque`.
///
/// A chunk is `[chunk, joints]`. `names` lists the model's joints in its
/// order (a subset of the robot's needs [`Allow::Select`]);
/// `positional(count)` takes the robot's first `count` by index.
pub fn joints() -> Joints {
    Joints(Draft::new("joints"))
}

/// Commands for a hand's joints: [`joints`] under its own kind, so it never
/// drives an arm.
pub fn hand() -> Hand {
    Hand(Draft::new("hand"))
}

/// A wrench to apply in `frame`: `force [3]` N and `torque [3]` N*m.
pub fn wrench() -> Wrench {
    Wrench(Draft::new("wrench"))
}

/// Pose targets for `child` in `frame`: absolute, or a delta from the
/// [`Reference`].
///
/// `rotation` is the encoding the model writes rotations in; another than
/// the part's needs [`Allow::Rotation`] unless it only reorders a
/// quaternion.
pub fn pose() -> Pose {
    Pose(Draft::new("pose"))
}

/// Gripper openings: `pos [1]`, 0 closed to 1 open.
///
/// `open` and `closed` are the model's values for each end (default 1 and
/// 0) when they differ from the robot's.
pub fn gripper() -> Gripper {
    Gripper(Draft::new("gripper"))
}

/// On/off targets: `on`, a bool per action.
pub fn switch() -> Switch {
    Switch(Draft::new("switch"))
}

/// Targets for one number: `value`, a float32 in the part's unit.
pub fn value() -> Value {
    Value(Draft::new("value"))
}

/// Base twists in `frame`: `linear` m/s and `angular` rad/s.
pub fn velocity() -> Velocity {
    Velocity(Draft::new("velocity"))
}

/// Commands for a custom kind `ns/name`, with its command fields as the
/// part declares them.
///
/// ```
/// use rlmesh_session_core::spec::actions::{self, Array, DType};
///
/// let mv = actions::custom("x/move")
///     .at("game:move")
///     .field("cell", Array::new(DType::I64, []))
///     .command("play");
/// ```
pub fn custom(kind: impl Into<String>) -> Custom {
    Custom(Draft::new(kind))
}
