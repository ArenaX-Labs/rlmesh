//! The kind table (§4): what each core kind publishes and accepts.
//!
//! | Kind | Published fields | Command modes → command fields | Default send | Class |
//! |---|---|---|---|---|
//! | `camera` | `image: u8 [H, W, 3]` (`mono8`: `[H, W, 1]`) | none | `latest(hz=hz)` | bulk |
//! | `depth` | `depth: f32 [H, W] m` or `u16 [H, W] mm` | none | `latest(hz=hz)` | bulk |
//! | `joints`, `hand` | `pos [N] rad`, `vel [N] rad/s`, `effort [N] N*m` (prismatic: m, m/s, N) | `position` → `pos`, `velocity` → `vel`, `torque` → `effort` | `every` | realtime |
//! | `wrench` | `force [3] N`, `torque [3] N*m` | `wrench` → both | `every` | realtime |
//! | `imu` | `orientation [R]`, `ang_vel [3] rad/s`, `lin_acc [3] m/s^2`, `gravity [3]` | none | `every` | realtime |
//! | `pose` | `pos [3] m`, `rot [R]` | `absolute` → `pos`, `rot`; `delta` → `pos [3] m`, `rot [3]` axis-angle | `every` | realtime |
//! | `gripper` | `pos [1]`, 0 closed … 1 open | `position` → `pos` | `every` | realtime |
//! | `velocity` | planar `linear [2] m/s`, `angular [1] rad/s`; spatial `[3]`, `[3]` | `velocity` → both | `every` | realtime |
//! | `tactile` | `values [shape]` normalized | none | `every` | data |
//! | `text`, `reward`, `success` | `text: utf8`, `reward: f32 []`, `success: bool []` | none | event | data |
//! | `ns/name` | declared | declared | `every` (≤ 4 kB), else `latest(hz)` | data or bulk |
//!
//! Float fields are `float32` unless the part declares `float64`. Shapes
//! have no batch or time axis.

use std::fmt;

use rlmesh_wire::path::is_identifier;

use crate::spec::units::Quantity;

/// The longest kind name, namespace or name.
const MAX_KIND_LEN: usize = 32;

/// A part kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// An RGB (or mono) image.
    Camera,
    /// A depth image.
    Depth,
    /// A set of joints, all revolute or all prismatic.
    Joints,
    /// A hand's joints: `joints` under another kind, so a hand policy cannot
    /// bind an arm.
    Hand,
    /// A force/torque sensor.
    Wrench,
    /// An inertial measurement unit.
    Imu,
    /// A pose of `child` in `frame`.
    Pose,
    /// A gripper opening.
    Gripper,
    /// A base twist.
    Velocity,
    /// Taxels.
    Tactile,
    /// Text (an event).
    Text,
    /// A reward (an event).
    Reward,
    /// An outcome (an event).
    Success,
    /// A namespaced custom kind (`ns/name`).
    Custom,
}

impl Kind {
    /// The core kinds.
    pub const CORE: [Self; 13] = [
        Self::Camera,
        Self::Depth,
        Self::Joints,
        Self::Hand,
        Self::Wrench,
        Self::Imu,
        Self::Pose,
        Self::Gripper,
        Self::Velocity,
        Self::Tactile,
        Self::Text,
        Self::Reward,
        Self::Success,
    ];

    /// Parses a kind: a core kind, or `ns/name` with `ns` other than
    /// `rlmesh` (both `[a-z][a-z0-9_]*`).
    pub fn parse(kind: &str) -> Result<Self, String> {
        match kind.split_once('/') {
            None => Self::CORE
                .into_iter()
                .find(|k| k.name() == kind)
                .ok_or_else(|| {
                    format!("kind {kind:?} is not a core kind (custom kinds are ns/name)")
                }),
            Some((ns, name)) => {
                if ns == "rlmesh" {
                    Err(format!("kind {kind:?}: the rlmesh/ namespace is reserved"))
                } else if is_identifier(ns, MAX_KIND_LEN) && is_identifier(name, MAX_KIND_LEN) {
                    Ok(Self::Custom)
                } else {
                    Err(format!("kind {kind:?}: ns/name are [a-z][a-z0-9_]*"))
                }
            }
        }
    }

    /// The core kind's name (`""` for custom kinds, whose name is the
    /// part's).
    pub fn name(self) -> &'static str {
        match self {
            Self::Camera => "camera",
            Self::Depth => "depth",
            Self::Joints => "joints",
            Self::Hand => "hand",
            Self::Wrench => "wrench",
            Self::Imu => "imu",
            Self::Pose => "pose",
            Self::Gripper => "gripper",
            Self::Velocity => "velocity",
            Self::Tactile => "tactile",
            Self::Text => "text",
            Self::Reward => "reward",
            Self::Success => "success",
            Self::Custom => "",
        }
    }

    /// An event kind: a bounded queue, reliable, never decimated (§8.5).
    pub fn is_event(self) -> bool {
        matches!(self, Self::Text | Self::Reward | Self::Success)
    }

    /// An image kind (camera or depth): the aspect-fit and resize
    /// conversions apply.
    pub fn is_image(self) -> bool {
        matches!(self, Self::Camera | Self::Depth)
    }

    /// A joint-set kind: names reorder and select.
    pub fn is_joint_set(self) -> bool {
        matches!(self, Self::Joints | Self::Hand)
    }

    /// The command modes a core kind accepts (custom kinds accept any).
    pub fn command_modes(self) -> &'static [&'static str] {
        match self {
            Self::Joints | Self::Hand => &["position", "velocity", "torque"],
            Self::Wrench => &["wrench"],
            Self::Pose => &["absolute", "delta"],
            Self::Gripper => &["position"],
            Self::Velocity => &["velocity"],
            _ => &[],
        }
    }

    /// Whether Hold may repeat the last dispatched target in `mode`
    /// (position-type modes, §9.5).
    pub fn holds_target(self, mode: &str) -> bool {
        matches!(
            (self, mode),
            (Self::Joints | Self::Hand | Self::Gripper, "position") | (Self::Pose, "absolute")
        )
    }

    /// The optional fields a joint set or IMU may publish, in payload order.
    pub fn optional_fields(self) -> &'static [&'static str] {
        match self {
            Self::Joints | Self::Hand => &["pos", "vel", "effort"],
            Self::Imu => &["orientation", "ang_vel", "lin_acc", "gravity"],
            _ => &[],
        }
    }

    /// What a policy needs from a part of this kind, for refusals ("this
    /// policy needs a force/torque sensor").
    pub fn need(self) -> &'static str {
        match self {
            Self::Camera => "a camera",
            Self::Depth => "a depth camera",
            Self::Joints => "a joints part",
            Self::Hand => "a hand",
            Self::Wrench => "a force/torque sensor",
            Self::Imu => "an IMU",
            Self::Pose => "a pose part",
            Self::Gripper => "a gripper",
            Self::Velocity => "a velocity part",
            Self::Tactile => "a tactile sensor",
            Self::Text => "a text part",
            Self::Reward => "a reward part",
            Self::Success => "a success part",
            Self::Custom => "a custom part",
        }
    }

    /// The plural noun for values of this kind ("base-frame wrenches").
    pub fn values_noun(self) -> &'static str {
        match self {
            Self::Wrench => "wrenches",
            Self::Imu => "IMU readings",
            Self::Pose => "poses",
            Self::Velocity => "velocities",
            Self::Camera | Self::Depth => "images",
            _ => "values",
        }
    }

    /// The singular noun ("publish the wrench in tool0").
    pub fn value_noun(self) -> &'static str {
        match self {
            Self::Wrench => "wrench",
            Self::Imu => "IMU reading",
            Self::Pose => "pose",
            Self::Velocity => "velocity",
            Self::Camera | Self::Depth => "image",
            _ => "value",
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if *self == Self::Custom {
            "custom"
        } else {
            self.name()
        })
    }
}

/// The quantity of field `field` of a part of `kind`, published in `unit`.
/// Joint fields are angular or linear by their unit (a prismatic set
/// publishes metres).
pub fn quantity(kind: Kind, field: &str, unit: &str) -> Quantity {
    let linear = matches!(unit, "m" | "mm" | "m/s" | "mm/s" | "N");
    match (kind, field) {
        (Kind::Joints | Kind::Hand, "pos") if linear => Quantity::Length,
        (Kind::Joints | Kind::Hand, "pos") => Quantity::Angle,
        (Kind::Joints | Kind::Hand, "vel") if linear => Quantity::LinearVelocity,
        (Kind::Joints | Kind::Hand, "vel") => Quantity::AngularVelocity,
        (Kind::Joints | Kind::Hand, "effort") if linear => Quantity::Force,
        (Kind::Joints | Kind::Hand, "effort") => Quantity::Torque,
        (Kind::Wrench, "force") => Quantity::Force,
        (Kind::Wrench, "torque") => Quantity::Torque,
        (Kind::Imu | Kind::Velocity, "ang_vel" | "angular") => Quantity::AngularVelocity,
        (Kind::Imu, "lin_acc") => Quantity::Acceleration,
        (Kind::Pose, "pos") => Quantity::Length,
        (Kind::Velocity, "linear") => Quantity::LinearVelocity,
        (Kind::Gripper, "pos") => Quantity::Gripper,
        (Kind::Depth, "depth") => Quantity::Length,
        (Kind::Tactile, "values") => Quantity::Tactile,
        (Kind::Custom, _) => Quantity::Custom,
        _ => Quantity::None,
    }
}

/// The quantity a joint field has, revolute or prismatic.
pub fn joint_quantity(field: &str, prismatic: bool) -> Quantity {
    match (field, prismatic) {
        ("pos", false) => Quantity::Angle,
        ("pos", true) => Quantity::Length,
        ("vel", false) => Quantity::AngularVelocity,
        ("vel", true) => Quantity::LinearVelocity,
        ("effort", false) => Quantity::Torque,
        ("effort", true) => Quantity::Force,
        _ => Quantity::None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_parse() {
        assert_eq!(Kind::parse("joints"), Ok(Kind::Joints));
        assert_eq!(Kind::parse("unitree/foot_force"), Ok(Kind::Custom));
        assert_eq!(Kind::parse("x/box"), Ok(Kind::Custom));
        for bad in ["rlmesh/x", "Camera", "points", "a/b/c", "/x", "x/"] {
            assert!(Kind::parse(bad).is_err(), "{bad}");
        }
        assert!(Kind::Pose.holds_target("absolute") && !Kind::Pose.holds_target("delta"));
        assert!(!Kind::Joints.holds_target("velocity"));
    }

    #[test]
    fn joint_quantities_follow_the_unit() {
        assert_eq!(quantity(Kind::Joints, "pos", "deg"), Quantity::Angle);
        assert_eq!(quantity(Kind::Joints, "pos", "mm"), Quantity::Length);
        assert_eq!(quantity(Kind::Hand, "effort", "N"), Quantity::Force);
        assert_eq!(quantity(Kind::Imu, "orientation", ""), Quantity::None);
    }
}
