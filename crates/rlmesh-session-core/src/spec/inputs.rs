//! A policy's inputs, built by kind: one constructor per kind, a setter per
//! key.
//!
//! Each builder stands for one input entry of the policy map, with only
//! the keys set; [`Policy::builder`](crate::spec::Policy::builder) takes it
//! beside entries written as JSON. The address is set with `at`
//! (`robot:path`, a path below the scope such as `wrist/cam`, or `/path`
//! from the robot's root; a segment may be `*` or a `{capture}`); without
//! one, the entry binds any part of its kind. Values the types cannot rule
//! out (a size of 0, an unknown frame name) are refused when the policy is
//! built, naming the entry.
//!
//! ```
//! use core::time::Duration;
//!
//! use rlmesh_session_core::spec::inputs::{self, Allow, DType, JointField};
//! use rlmesh_session_core::spec::Policy;
//!
//! let policy = Policy::builder()
//!     .hz(15)
//!     .input(
//!         "image",
//!         inputs::camera()
//!             .at("hallo-cam:cam")
//!             .size(224)
//!             .dtype(DType::F32)
//!             .range(0, 1)
//!             .allow([Allow::Resize]),
//!     )
//!     .input(
//!         "state",
//!         inputs::joints()
//!             .at("arm")
//!             .fields([JointField::Pos])
//!             .unit("pos", "deg")
//!             .max_age(Duration::from_millis(200)),
//!     )
//!     .input("task", inputs::text().optional(true))
//!     .build()?;
//! assert_eq!(policy.inputs[0].size, Some((224, 224)));
//! # Ok::<(), String>(())
//! ```
//!
//! Every input takes `optional` (may stay unbound, with `blank` allowed),
//! `max_age` (the oldest sample accepted at a start) and `allow`. A
//! streamed kind also takes `history` and `history_hz`; a commandable kind
//! takes `source(Source::Applied)` to read what its edge dispatched;
//! numeric kinds take `dtype`, `unit`, `scale`, `offset` and `range`.

#[allow(unused_imports)]
use crate::prelude::*;

use serde_json::Value as Json;

use crate::spec::policy::{Draft, Val, builder, named};

pub use crate::spec::Rotation;
pub use crate::spec::policy::{
    Allow, Array, ImageFormat, ImageLayout, PerElement, Resample, Source,
};
pub use crate::wire::part::DType;

/// An image size: `(height, width)`, or one side for a square.
///
/// `224`, `(96, 128)` and `[96, 128]` convert into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    /// Rows.
    pub height: u32,
    /// Columns.
    pub width: u32,
}

impl From<u32> for ImageSize {
    fn from(side: u32) -> Self {
        Self {
            height: side,
            width: side,
        }
    }
}

impl From<(u32, u32)> for ImageSize {
    fn from((height, width): (u32, u32)) -> Self {
        Self { height, width }
    }
}

impl From<[u32; 2]> for ImageSize {
    fn from([height, width]: [u32; 2]) -> Self {
        Self { height, width }
    }
}

named! {
    /// A joint set's field.
    JointField {
        /// Positions.
        Pos = "pos",
        /// Velocities.
        Vel = "vel",
        /// Efforts.
        Effort = "effort",
    }
}

named! {
    /// An IMU's field.
    ImuField {
        /// Orientation, in the entry's rotation encoding.
        Orientation = "orientation",
        /// Angular velocity.
        AngVel = "ang_vel",
        /// Linear acceleration.
        LinAcc = "lin_acc",
        /// The gravity direction.
        Gravity = "gravity",
    }
}

named! {
    /// A wrench's field.
    WrenchField {
        /// Force, N.
        Force = "force",
        /// Torque, N*m.
        Torque = "torque",
    }
}

named! {
    /// A pose's field.
    PoseField {
        /// Position, m.
        Pos = "pos",
        /// Rotation, in the entry's rotation encoding.
        Rot = "rot",
    }
}

named! {
    /// A twist's field.
    VelocityField {
        /// Linear velocity, m/s.
        Linear = "linear",
        /// Angular velocity, rad/s.
        Angular = "angular",
    }
}

builder! {
    /// A camera input; see [`camera`].
    Camera => InputEntry [
        size, layout, format, resample, crop, upside_down, dtype, range, frame,
        optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A depth camera input; see [`depth`].
    Depth => InputEntry [
        size, crop, upside_down, dtype, unit, range, frame,
        optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A joint set input; see [`joints`].
    Joints => InputEntry [
        names, positional, count, fields(JointField), dtype, unit, scale, offset, range,
        source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A hand input; see [`hand`].
    Hand => InputEntry [
        names, positional, count, fields(JointField), dtype, unit, scale, offset, range,
        source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A force/torque input; see [`wrench`].
    Wrench => InputEntry [
        frame, point, fields(WrenchField), dtype, unit, scale, offset, range,
        source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// An IMU input; see [`imu`].
    Imu => InputEntry [
        frame, fields(ImuField), rotation, dtype, unit, scale, offset,
        optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A pose input; see [`pose`].
    Pose => InputEntry [
        frame, child, fields(PoseField), rotation, dtype, unit, scale, offset,
        source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A gripper input; see [`gripper`].
    Gripper => InputEntry [
        dtype, scale, offset, range, source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A switch input; see [`switch`].
    Switch => InputEntry [dtype, source, optional, max_age, history, history_hz, allow]
}

builder! {
    /// A value input; see [`value`].
    Value => InputEntry [
        dtype, unit, scale, offset, range, source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A base twist input; see [`velocity`].
    Velocity => InputEntry [
        frame, fields(VelocityField), dtype, unit, scale, offset, range,
        source, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// A tactile input; see [`tactile`].
    Tactile => InputEntry [
        shape, dtype, scale, offset, range, optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// An event input (text, reward or success); see [`text`], [`reward`]
    /// and [`success`].
    Event => InputEntry [optional, max_age, allow]
}

builder! {
    /// A custom kind's input; see [`custom`].
    Custom => InputEntry [
        field, frame, unit, scale, offset, range, source,
        optional, max_age, history, history_hz, allow,
    ]
}

builder! {
    /// An input reading a declared part; see [`of`]. It takes the setters
    /// of the part's kind: another is refused when the policy is built.
    Of => InputEntry [
        size, layout, format, resample, crop, upside_down, dtype, range, frame, child, point,
        names, positional, count, unit, scale, offset, rotation, shape, source,
        optional, max_age, history, history_hz, allow,
    ]
}

/// A camera's frames: `uint8 [height, width, 3]` (`Hwc`, `Rgb8`) by
/// default.
///
/// `size` resizes (with [`Allow::Resize`]); `crop` is the side fraction of
/// a center crop the model was trained with; `upside_down` a model trained
/// on frames rotated 180°. `dtype` with `range(0, 1)` normalises the
/// pixels.
pub fn camera() -> Camera {
    Camera(Draft::new("camera"))
}

/// A depth camera's frames: `depth: float32 [height, width]` in m.
///
/// `size` resizes by nearest neighbour; `unit("depth", "mm")` asks for
/// millimetres.
pub fn depth() -> Depth {
    Depth(Draft::new("depth"))
}

/// A set of joints: the fields `pos`, `vel` and `effort` (default: all the
/// part publishes).
///
/// `names` lists the model's joints in its order (a subset of the robot's
/// needs [`Allow::Select`]); `positional(count)` takes the robot's first
/// `count` by index.
pub fn joints() -> Joints {
    Joints(Draft::new("joints"))
}

/// A hand's joints: [`joints`] under its own kind, so it never binds an
/// arm.
pub fn hand() -> Hand {
    Hand(Draft::new("hand"))
}

/// A force/torque reading: `force [3]` N and `torque [3]` N*m.
///
/// `frame` and `point` are the frame and reference point the model needs;
/// a part declared in another refuses to bind.
pub fn wrench() -> Wrench {
    Wrench(Draft::new("wrench"))
}

/// An IMU reading: the fields `orientation`, `ang_vel`, `lin_acc` and
/// `gravity`.
///
/// `rotation` is the encoding the model takes the orientation in
/// (`GravityXyz` too); another than the part's needs [`Allow::Rotation`]
/// unless it only reorders a quaternion.
pub fn imu() -> Imu {
    Imu(Draft::new("imu"))
}

/// The pose of `child` in `frame`: `pos [3]` m and `rot` in `rotation`.
pub fn pose() -> Pose {
    Pose(Draft::new("pose"))
}

/// A gripper's opening: `pos [1]`, 0 closed to 1 open (`range` with
/// [`Allow::Rescale`] maps it).
pub fn gripper() -> Gripper {
    Gripper(Draft::new("gripper"))
}

/// An on/off part's state: `on`, a bool (`dtype` with [`Allow::Dtype`]
/// casts it).
pub fn switch() -> Switch {
    Switch(Draft::new("switch"))
}

/// One number: `value`, a float32 in the part's unit (`unit("value", ...)`
/// asks for another).
pub fn value() -> Value {
    Value(Draft::new("value"))
}

/// A base twist in `frame`: `linear` m/s and `angular` rad/s.
pub fn velocity() -> Velocity {
    Velocity(Draft::new("velocity"))
}

/// Taxels: `values`, normalized; `shape` is the one the model expects.
pub fn tactile() -> Tactile {
    Tactile(Draft::new("tactile"))
}

/// Text events: the newest string (a task, a label).
pub fn text() -> Event {
    Event(Draft::new("text"))
}

/// Reward events: the newest `float32 []`.
pub fn reward() -> Event {
    Event(Draft::new("reward"))
}

/// Outcome events: the newest `bool []`.
pub fn success() -> Event {
    Event(Draft::new("success"))
}

/// A custom kind `ns/name`, read with its fields as the part declares them.
///
/// ```
/// use rlmesh_session_core::spec::inputs::{self, Array, DType};
///
/// let boxes = inputs::custom("yolo/boxes")
///     .at("tracker:boxes")
///     .field("xy", Array::new(DType::F32, [8, 2]));
/// ```
pub fn custom(kind: impl Into<String>) -> Custom {
    Custom(Draft::new(kind))
}

/// An input reading `part`, one part of a robot map as JSON (such as a
/// policy's own published part): its kind copied, and its fields when the
/// kind declares them (`joints`, `hand`, `imu` and custom kinds).
///
/// ```
/// use rlmesh_session_core::spec::inputs;
///
/// let pose = serde_json::json!({
///     "kind": "pose/coco17", "hz": 30, "fields": {"kp": ["float32", [17, 3]]}
/// });
/// let reader = inputs::of(&pose).at("tracker:pose").optional(true);
/// ```
pub fn of(part: &Json) -> Of {
    let Some(kind) = part.get("kind").and_then(Json::as_str) else {
        let problem = format!("of: {part} is not a part: it has no \"kind\"");
        return Of(Draft::new("").refused(problem));
    };
    let mut draft = Draft::new(kind);
    let copies_fields = matches!(kind, "joints" | "hand" | "imu") || kind.contains('/');
    if copies_fields && let Some(fields) = part.get("fields") {
        draft.set("fields", Val::Json(fields.clone()));
    }
    let takes = match kind {
        "camera" => Camera::KEYS,
        "depth" => Depth::KEYS,
        "joints" => Joints::KEYS,
        "hand" => Hand::KEYS,
        "wrench" => Wrench::KEYS,
        "imu" => Imu::KEYS,
        "pose" => Pose::KEYS,
        "gripper" => Gripper::KEYS,
        "switch" => Switch::KEYS,
        "value" => Value::KEYS,
        "velocity" => Velocity::KEYS,
        "tactile" => Tactile::KEYS,
        "text" | "reward" | "success" => Event::KEYS,
        _ if kind.contains('/') => Custom::KEYS,
        // Not a kind: the normaliser refuses it.
        _ => return Of(draft),
    };
    Of(draft.taking(takes))
}
