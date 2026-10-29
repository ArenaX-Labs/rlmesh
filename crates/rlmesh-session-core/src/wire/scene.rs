//! The lobby and scenes (scene design §3, §4, §5): a waiting participant's
//! assignment, the orders a home daemon sends satellites, a run's state, and
//! the places and scene a session's authority state carries.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::wire::authority::{CtlError, PriorityClass, Reason};
use crate::wire::ids::{Fingerprint, ParticipantId, SessionId};

/// A waiting robot or satellite asks the home daemon for its next order
/// (a long poll).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LobbyWait {
    /// The run that just failed for this participant (empty: none).
    pub run: String,
    /// Why it failed.
    pub last: Option<CtlError>,
}

/// The answer to a [`LobbyWait`].
#[derive(Clone, Debug, PartialEq)]
pub enum Assignment {
    /// A robot's: join a run's session.
    Join(Join),
    /// A satellite's: host a scene's session.
    Host(HostOrder),
    /// A satellite's: serve a model as a scene member.
    Serve(ServeOrder),
    /// A satellite's: close what it runs for a run.
    Stop(StopOrder),
}

/// A robot's order: join `session` at `addresses`, with an IK handshake
/// against `host_key`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Join {
    /// The run.
    pub run: String,
    /// Its session.
    pub session: SessionId,
    /// The robot it is for: the receiver refuses another name.
    pub robot: ParticipantId,
    /// The scene host's base addresses (`host:port`, class i at base + i).
    pub addresses: Vec<String>,
    /// The scene host's static X25519 key.
    pub host_key: [u8; 32],
    /// The scene host's id.
    pub host_id: String,
    /// The handshake prologue's tenant.
    pub tenant: String,
}

/// A satellite's order to host a scene's session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostOrder {
    /// The run.
    pub run: String,
    /// Its session.
    pub session: SessionId,
    /// The scene file.
    pub scene_toml: String,
    /// The members' devices.
    pub devices: Vec<DeviceRecord>,
    /// The robots' pins.
    pub pins: Vec<RobotPin>,
}

/// A satellite's order to serve a catalog model as a scene member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeOrder {
    /// The run.
    pub run: String,
    /// The member it serves as.
    pub member: ParticipantId,
    /// The catalog model (`prebuilt/smolvla/so100`).
    pub model: String,
    /// The run's session.
    pub session: SessionId,
    /// The scene host's base addresses.
    pub addresses: Vec<String>,
    /// The scene host's static X25519 key.
    pub host_key: [u8; 32],
}

/// A satellite's order to close what it runs for `run`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StopOrder {
    /// The run.
    pub run: String,
    /// Why ([`crate::wire::ReasonCode::SceneEnded`] or `SceneFailed`).
    pub reason: Reason,
}

/// A satellite's answer to a [`HostOrder`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostStarted {
    /// The run.
    pub run: String,
    /// The scene host's static key and base addresses, or why it didn't
    /// start.
    pub result: Result<([u8; 32], Vec<String>), CtlError>,
}

/// A member's device, as the home daemon's table has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRecord {
    /// Its id.
    pub id: String,
    /// Its static X25519 key.
    pub public_key: [u8; 32],
    /// Its cap (`None`: observer only).
    pub max_priority: Option<PriorityClass>,
    /// Globs over `{robot}/{path}` it may read.
    pub parts: Vec<String>,
    /// The verbs it may send (`arm`, `disarm`).
    pub verbs: Vec<String>,
}

/// A robot name's pin: the device that embodies it and, for a tethered
/// board, its board id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RobotPin {
    /// The robot name.
    pub robot: ParticipantId,
    /// Its device.
    pub device: String,
    /// The tethered board's id (tether spec §7).
    pub board_id: Option<[u8; 8]>,
}

/// Where a run is (scene design §4.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RunStage {
    /// Checking the scene against devices and pins.
    Checking,
    /// Starting its session host.
    Hosting,
    /// Assigning robots and serve members.
    Assigning,
    /// Waiting for robots' declarations.
    Declaring,
    /// Waiting for controllers' bindings.
    Binding,
    /// Waiting for (or sending) the arm.
    Arming,
    /// Running.
    Running,
    /// Tearing down.
    Ending,
    /// Ended (a stop).
    Ended,
    /// Failed.
    Failed,
    /// Waiting for a lost member (§4.3).
    Rejoining,
}

/// Where one member of a run is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemberStage {
    /// Not in the lobby yet.
    Waiting,
    /// Sent its order.
    Assigned,
    /// Attached to the run's session.
    Joined,
    /// Declared its parts.
    Declared,
    /// Declared its binding.
    Bound,
    /// Lost from the run.
    Lost,
    /// Back in the lobby.
    Returned,
}

/// One member's state in a run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberState {
    /// The member name.
    pub name: ParticipantId,
    /// Its stage.
    pub stage: MemberStage,
    /// Its last refusal.
    pub error: Option<CtlError>,
}

/// A run's state, as `RunScene`, `SceneStatus`, `StopScene` and `ArmScene`
/// answer it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunState {
    /// The run id.
    pub run: String,
    /// Rises with every change.
    pub version: u64,
    /// Its stage.
    pub stage: RunStage,
    /// Its members.
    pub members: Vec<MemberState>,
    /// Why it failed.
    pub error: Option<CtlError>,
    /// The scene host's session, once it is up.
    pub session: Option<SessionId>,
    /// The scene host's base addresses.
    pub addresses: Vec<String>,
    /// The scene host's static key, once it is up.
    pub host_key: Option<[u8; 32]>,
    /// The scene fingerprint, once checked.
    pub scene_fingerprint: Option<Fingerprint>,
}

/// One robot's place in a scene's view (§5).
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    /// The robot.
    pub robot: ParticipantId,
    /// Where its parts appear in the view.
    pub at: String,
    /// The path in its map placed there (empty: its root).
    pub from: String,
    /// The frame `pose` is the pose of (empty: `base`).
    pub frame: String,
    /// That frame's static pose in the scene frame `world`.
    pub pose: Pose,
}

/// A static pose: metres, and a unit quaternion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    /// Position.
    pub xyz: [f64; 3],
    /// Orientation (x, y, z, w).
    pub quat_xyzw: [f64; 4],
}

/// The scene a session runs, when a daemon started it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SceneInfo {
    /// The run id.
    pub run: String,
    /// The scene's name.
    pub name: String,
    /// The scene fingerprint.
    pub fingerprint: Fingerprint,
    /// Set once the run is ending: robots detach and return to the lobby.
    pub closing: Option<Reason>,
    /// Host session clock when the rejoin window closes, while a member is
    /// lost.
    pub rejoin_until_ns: Option<i64>,
}
