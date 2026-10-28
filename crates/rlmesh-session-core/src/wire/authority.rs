//! Authority state and control verbs: roles, the session mode, leases bound
//! to bindings, clock mappings, the host's snapshot, and the `ctl`
//! request/reply pair (§10.5).

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::BTreeMap;
use core::fmt;

use crate::wire::action::ChunkRef;
use crate::wire::contract::{ActionGrid, Demand, StreamContract};
use crate::wire::ids::{ChannelId, Epoch, Fingerprint, LeaseToken, ParticipantId, SessionId};
use crate::wire::key::{IssuedKey, KeysAck};
use crate::wire::part::{PartSpec, StreamSpec};

/// What a participant does in a session. One participant may hold several.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Publishes a robot's parts and consumes their commands (real, sim or
    /// twin).
    Embodiment,
    /// Produces commands (model replica, operator or scripted controller).
    Controller,
    /// Only subscribes (twin renderer, recorder, dashboard).
    Observer,
}

/// The session mode, owned by the host and enforced by every edge.
///
/// ```text
/// DISARMED ──arm──▶ ARMED ──grant lease──▶ ACTIVE ──lease lapse──▶ SAFE
///     ▲               ▲  ◀──release last lease─┘                    │
///     └──── disarm ───┴───────────── arm (fresh epoch) ─────────────┘
/// ANY ──estop──▶ ESTOP (latched; leaving needs local clear, disarm, arm)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// Nothing may move.
    Disarmed,
    /// Ready: a lease grant goes ACTIVE.
    Armed,
    /// At least one lease is held and commands flow.
    Active,
    /// A fallback happened; re-arming (a fresh epoch) is the way back.
    Safe,
    /// Emergency stop.
    Estop,
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Disarmed => "DISARMED",
            Self::Armed => "ARMED",
            Self::Active => "ACTIVE",
            Self::Safe => "SAFE",
            Self::Estop => "ESTOP",
        })
    }
}

/// Lease arbitration class, lowest first: a higher class preempts a lower
/// one (`safety > operator > model > scripted`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PriorityClass {
    /// Scripted controllers.
    Scripted,
    /// Model replicas.
    Model,
    /// Human operators (jump-in teleop).
    Operator,
    /// Safety supervisors.
    Safety,
}

/// How session time advances (§11.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ClockMode {
    /// The host's monotonic clock plus per-participant mappings.
    #[default]
    Wall,
    /// `tick × tick_ns`, advanced by the host tick by tick (§11.3).
    Virtual,
    /// `step × tick_ns`, advanced by the lockstep exchange (§12).
    Lockstep,
}

/// Why a mode changed, an edge or the host went SAFE, or a lease ended: one
/// code for all three.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ReasonCode {
    /// None given.
    #[default]
    Unspecified,
    /// Armed.
    Armed,
    /// Disarmed.
    Disarmed,
    /// The first lease went ACTIVE.
    FirstLease,
    /// The last lease ended: back to ARMED.
    LastRelease,
    /// A `ctl` `Estop` through the host.
    EstopRemote,
    /// `robot.estop()` or a hardware input at the edge.
    EstopLocal,
    /// An e-stop signal latched at the edge.
    EstopSignal,
    /// The embodiment cleared the latch.
    EstopCleared,
    /// The lease holder's heartbeats lapsed.
    HolderLost,
    /// The holder was silent past its lease's ttl.
    HeartbeatLapse,
    /// Edge: no valid action within `hold_max`.
    QueueDry,
    /// Edge: the lease expired by the edge's clock.
    EdgeLeaseLapsed,
    /// Edge: a group handover took longer than `hold_max`.
    HandoverTimeout,
    /// A device watchdog (a tethered board's link) fired.
    DeviceWatchdog,
    /// An embodiment's heartbeats lapsed.
    RobotLost,
    /// The session host is gone (participant side only).
    HostLost,
    /// The holder released the lease.
    Released,
    /// A higher class took it.
    Preempted,
    /// Re-issued under a new binding.
    Rebound,
    /// Ended when the session went SAFE.
    VoidedBySafe,
    /// Ended by an e-stop.
    VoidedByEstop,
    /// Ended by a disarm.
    VoidedByDisarm,
    /// Ended by a new epoch (arm, reset).
    VoidedByEpoch,
    /// Its part was removed.
    PartRemoved,
    /// Its holder detached.
    HolderDetached,
    /// The owner revoked the holder's device.
    Revoked,
    /// The owner changed the holder's device entry.
    AccessChanged,
}

impl ReasonCode {
    /// The name, without the wire's `REASON_` prefix (`ROBOT_LOST`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Unspecified => "UNSPECIFIED",
            Self::Armed => "ARMED",
            Self::Disarmed => "DISARMED",
            Self::FirstLease => "FIRST_LEASE",
            Self::LastRelease => "LAST_RELEASE",
            Self::EstopRemote => "ESTOP_REMOTE",
            Self::EstopLocal => "ESTOP_LOCAL",
            Self::EstopSignal => "ESTOP_SIGNAL",
            Self::EstopCleared => "ESTOP_CLEARED",
            Self::HolderLost => "HOLDER_LOST",
            Self::HeartbeatLapse => "HEARTBEAT_LAPSE",
            Self::QueueDry => "QUEUE_DRY",
            Self::EdgeLeaseLapsed => "EDGE_LEASE_LAPSED",
            Self::HandoverTimeout => "HANDOVER_TIMEOUT",
            Self::DeviceWatchdog => "DEVICE_WATCHDOG",
            Self::RobotLost => "ROBOT_LOST",
            Self::HostLost => "HOST_LOST",
            Self::Released => "RELEASED",
            Self::Preempted => "PREEMPTED",
            Self::Rebound => "REBOUND",
            Self::VoidedBySafe => "VOIDED_BY_SAFE",
            Self::VoidedByEstop => "VOIDED_BY_ESTOP",
            Self::VoidedByDisarm => "VOIDED_BY_DISARM",
            Self::VoidedByEpoch => "VOIDED_BY_EPOCH",
            Self::PartRemoved => "PART_REMOVED",
            Self::HolderDetached => "HOLDER_DETACHED",
            Self::Revoked => "REVOKED",
            Self::AccessChanged => "ACCESS_CHANGED",
        }
    }
}

impl fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A structured reason: the code, who caused it and when.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Reason {
    /// What happened.
    pub code: ReasonCode,
    /// The participant that caused it (empty: none).
    pub by: String,
    /// A human detail (empty: none).
    pub detail: String,
    /// Host session clock (the participant's clock for
    /// [`ReasonCode::HostLost`]).
    pub at_ns: i64,
    /// The revision the change landed in.
    pub revision: u64,
    /// The host authenticated the sender itself.
    pub verified: bool,
}

impl Reason {
    /// A reason with `code` and nothing else.
    pub fn new(code: ReasonCode) -> Self {
        Self {
            code,
            ..Self::default()
        }
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code)?;
        if !self.by.is_empty() {
            write!(f, " by {}", self.by)?;
        }
        if !self.detail.is_empty() {
            write!(f, ": {}", self.detail)?;
        }
        Ok(())
    }
}

/// One lease that ended, as the snapshot's ring keeps it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeaseEnd {
    /// The lease.
    pub lease: Lease,
    /// Why.
    pub reason: Reason,
    /// The lease that preempted or re-issued it.
    pub by: Option<Lease>,
}

/// The host's last entry into SAFE, DISARMED or ESTOP. Never lowered: a
/// snapshot showing `epoch ≥ E` proves the host stopped at or after epoch E.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LastStop {
    /// The epoch it happened in.
    pub epoch: Epoch,
    /// The revision it landed in.
    pub revision: u64,
}

/// One roster entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParticipantInfo {
    /// Its id.
    pub id: ParticipantId,
    /// A display name.
    pub name: String,
    /// Its roles.
    pub roles: Vec<Role>,
    /// The class its lease requests are arbitrated at.
    pub priority: PriorityClass,
    /// Host session clock when it (last) attached.
    pub attached_ns: i64,
    /// The protocol minor it announced.
    pub protocol_minor: u32,
    /// Features it implements.
    pub supports: Vec<String>,
    /// Features it cannot run without.
    pub requires: Vec<String>,
    /// The workflow edition it declares (empty: none).
    pub workflow_edition: String,
    /// Every workflow edition it runs.
    pub workflow_editions: Vec<String>,
    /// Its heartbeats arrive. A robot whose heartbeats lapsed stays in the
    /// roster, absent, while its parts remain.
    pub present: bool,
    /// Why it is absent ([`ReasonCode::RobotLost`]).
    pub left: Option<Reason>,
    /// The device its handshake authenticated.
    pub device: String,
    /// The process attached under this id: rises with every handshake that
    /// takes effect.
    pub generation: u32,
}

impl ParticipantInfo {
    /// Whether it holds `role`.
    pub fn has_role(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }
}

/// Command authority over one commandable part, bound to one binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lease {
    /// The commandable part.
    pub channel: ChannelId,
    /// The controller holding it.
    pub holder: ParticipantId,
    /// The holder's class.
    pub priority: PriorityClass,
    /// The fencing token chunks must carry.
    pub token: LeaseToken,
    /// Each heartbeat from the holder pushes `expires_ns` to arrival + ttl.
    pub ttl_ns: u64,
    /// Host session clock at the grant.
    pub granted_ns: i64,
    /// Host session clock at which it lapses without a renewal.
    pub expires_ns: i64,
    /// The binding every chunk under it must carry.
    pub binding: Fingerprint,
    /// The id of its `k_cmd` (§8.4): never the key itself.
    pub key_id: u32,
}

/// A published clock mapping: `session_ns = participant_ns + offset_ns`,
/// within `± error_ns`.
#[derive(Clone, Debug, PartialEq)]
pub struct ClockMapping {
    /// Whose clock.
    pub participant: ParticipantId,
    /// Session minus participant.
    pub offset_ns: i64,
    /// The error bound as of the snapshot.
    pub error_ns: u64,
    /// The round trip of the sample the offset came from.
    pub rtt_ns: u64,
    /// Host session clock of that sample.
    pub sample_ns: i64,
    /// The drift bound the error grows by with the sample's age.
    pub drift_bound_ppm: f64,
}

impl ClockMapping {
    /// Converts one of the participant's timestamps into session time, with
    /// the error bound as of the snapshot.
    pub fn to_session_ns(&self, participant_ns: i64) -> (i64, u64) {
        (participant_ns.saturating_add(self.offset_ns), self.error_ns)
    }
}

/// A controller's binding, as the host recorded it (§6.8, §9.7). The binder
/// (`rlmesh_session_core::spec`) produces the table and its fingerprint; the host keeps
/// what fencing and scheduling need.
#[derive(Clone, Debug, PartialEq)]
pub struct BindingRecord {
    /// The controller.
    pub participant: ParticipantId,
    /// The fingerprint its leases are bound to.
    pub fingerprint: Fingerprint,
    /// The controller's revision counter.
    pub revision: u32,
    /// The binding table, as JSON (opaque to the host).
    pub table_json: String,
    /// The commandable parts it may lease.
    pub channels: Vec<ChannelId>,
    /// Its replan rate R; in virtual mode, the ticks it is due on (0: never
    /// due).
    pub replan_hz: f64,
    /// Its delivery contracts, one per bound input stream (§8.2).
    pub contracts: Vec<StreamContract>,
    /// Its action grids, one per commanded part (§9.2).
    pub grids: Vec<ActionGrid>,
}

impl BindingRecord {
    /// The action period it declared for `channel`.
    pub fn grid(&self, channel: ChannelId) -> Option<core::time::Duration> {
        self.grids
            .iter()
            .find(|g| g.channel == channel)
            .map(|g| g.dt)
    }
}

/// A chunk (or verb) failed the epoch/lease fence.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FenceError {
    /// From another epoch.
    #[error("epoch {got} is not the current {current}")]
    StaleEpoch {
        /// The current epoch.
        current: Epoch,
        /// The one it carried.
        got: Epoch,
    },
    /// The part has no lease.
    #[error("no lease on {channel}")]
    NoLease {
        /// The part.
        channel: ChannelId,
    },
    /// Fenced: not the current lease's token (a preempted holder).
    #[error("token {got} on {channel} is fenced; the current lease is {current}")]
    StaleToken {
        /// The part.
        channel: ChannelId,
        /// The current token.
        current: LeaseToken,
        /// The one it carried.
        got: LeaseToken,
    },
    /// The lease expired.
    #[error("the lease on {channel} expired")]
    Expired {
        /// The part.
        channel: ChannelId,
    },
}

/// The host's authority state at one instant: what `host/state` serves.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthoritySnapshot {
    /// The session.
    pub session: SessionId,
    /// The current epoch.
    pub epoch: Epoch,
    /// Bumps on every structural change.
    pub revision: u64,
    /// The mode.
    pub mode: Mode,
    /// Whether an e-stop is latched.
    pub estop_latched: bool,
    /// Attached participants.
    pub roster: Vec<ParticipantInfo>,
    /// Held leases.
    pub leases: Vec<Lease>,
    /// Clock mappings per participant.
    pub clocks: Vec<ClockMapping>,
    /// Host session clock when the snapshot was taken.
    pub host_ns: i64,
    /// Every declared part.
    pub parts: Vec<PartSpec>,
    /// The host's protocol minor.
    pub protocol_minor: u32,
    /// The features the host implements.
    pub supports: Vec<String>,
    /// How session time advances.
    pub clock_mode: ClockMode,
    /// Session time per tick (virtual and lockstep modes; 0 in wall mode).
    pub tick_ns: u64,
    /// Every controller's binding.
    pub bindings: Vec<BindingRecord>,
    /// Each robot's declaration fingerprint (§3.3).
    pub contract: BTreeMap<ParticipantId, Fingerprint>,
    /// The publication plan of every stream some binding has a contract on.
    pub demands: Vec<Demand>,
    /// Why the mode is what it is.
    pub mode_reason: Reason,
    /// The latest lease endings, oldest first.
    pub ended: Vec<LeaseEnd>,
    /// The last entry into SAFE, DISARMED or ESTOP.
    pub last_stop: LastStop,
}

impl AuthoritySnapshot {
    /// The part whose stream is `channel`.
    pub fn part(&self, channel: ChannelId) -> Option<&PartSpec> {
        self.parts.iter().find(|p| p.channel == channel)
    }

    /// The part at `path` of `robot`.
    pub fn part_at(&self, robot: &ParticipantId, path: &str) -> Option<&PartSpec> {
        self.parts
            .iter()
            .find(|p| &p.owner == robot && p.path.as_str() == path)
    }

    /// The stream `channel` names: a part's samples or its applied stream.
    pub fn stream(&self, channel: ChannelId) -> Option<StreamSpec> {
        self.parts.iter().find_map(|p| {
            if p.channel == channel {
                p.stream().ok()
            } else if p.applied_channel == Some(channel) {
                p.applied_stream().and_then(Result::ok)
            } else {
                None
            }
        })
    }

    /// The commandable parts of `robot` in command group `group`.
    pub fn group_members<'a>(
        &'a self,
        robot: &'a ParticipantId,
        group: &'a str,
    ) -> impl Iterator<Item = &'a PartSpec> + 'a {
        self.parts.iter().filter(move |p| {
            &p.owner == robot
                && p.command
                    .as_ref()
                    .is_some_and(|c| c.group.as_deref() == Some(group))
        })
    }

    /// The lease on `channel`.
    pub fn lease(&self, channel: ChannelId) -> Option<&Lease> {
        self.leases.iter().find(|lease| lease.channel == channel)
    }

    /// A roster entry.
    pub fn participant(&self, id: &ParticipantId) -> Option<&ParticipantInfo> {
        self.roster.iter().find(|p| &p.id == id)
    }

    /// A participant's clock mapping.
    pub fn clock(&self, id: &ParticipantId) -> Option<&ClockMapping> {
        self.clocks.iter().find(|c| &c.participant == id)
    }

    /// A controller's binding.
    pub fn binding(&self, id: &ParticipantId) -> Option<&BindingRecord> {
        self.bindings.iter().find(|b| &b.participant == id)
    }

    /// The publication plan of stream `channel`.
    pub fn demand(&self, channel: ChannelId) -> Option<&Demand> {
        self.demands.iter().find(|d| d.channel == channel)
    }

    /// The fence an edge applies to an incoming chunk: right epoch, and the
    /// token of the part's current lease. The edge checks expiry itself,
    /// against its own clock.
    pub fn check_fence(
        &self,
        channel: ChannelId,
        token: LeaseToken,
        epoch: Epoch,
    ) -> Result<&Lease, FenceError> {
        fence(self.epoch, self.lease(channel), channel, token, epoch)
    }

    /// Orders snapshots from one host: a later revision, or a later
    /// republish of the same revision (fresher clock mappings).
    pub fn is_newer_than(&self, other: &Self) -> bool {
        (self.revision, self.host_ns) > (other.revision, other.host_ns)
    }
}

/// The epoch-and-token fence, shared by the host and the edges.
pub fn fence(
    current_epoch: Epoch,
    lease: Option<&Lease>,
    channel: ChannelId,
    token: LeaseToken,
    epoch: Epoch,
) -> Result<&Lease, FenceError> {
    if epoch != current_epoch {
        return Err(FenceError::StaleEpoch {
            current: current_epoch,
            got: epoch,
        });
    }
    let lease = lease.ok_or(FenceError::NoLease { channel })?;
    if lease.token != token {
        return Err(FenceError::StaleToken {
            channel,
            current: lease.token,
            got: token,
        });
    }
    Ok(lease)
}

/// The last seq of one stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Watermark {
    /// The stream.
    pub channel: ChannelId,
    /// Its last seq.
    pub seq: u64,
}

/// A control verb.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Join the roster (or re-join, keeping held leases).
    Attach {
        /// A display name.
        name: String,
        /// Its roles; at least one.
        roles: Vec<Role>,
        /// Its lease class.
        priority: PriorityClass,
        /// The protocol minor it speaks.
        protocol_minor: u32,
        /// Features it implements.
        supports: Vec<String>,
        /// Features it cannot run without (refused with
        /// [`CtlStatus::Incompatible`] when the host lacks one).
        requires: Vec<String>,
        /// The workflow edition it declares (empty: none).
        workflow_edition: String,
        /// Every workflow edition it runs.
        workflow_editions: Vec<String>,
    },
    /// Leave, releasing leases and removing owned parts.
    Detach,
    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    Arm,
    /// Back to DISARMED, revoking every lease.
    Disarm,
    /// An embodiment's robot map, as authored JSON; the host normalises it.
    DeclareParts {
        /// The kind table version: [`crate::wire::version::SPEC_VERSION`].
        spec_version: u32,
        /// The authored map.
        robot_json: String,
        /// The part paths this robot process can seal (§8.8).
        seal: Vec<String>,
    },
    /// A controller's binding. A changed fingerprint voids the controller's
    /// leases and re-issues them under fresh tokens.
    DeclareBinding {
        /// The binding's fingerprint.
        fingerprint: Fingerprint,
        /// The controller's revision counter.
        revision: u32,
        /// The binding table, as JSON.
        table_json: String,
        /// The commandable parts it will lease.
        channels: Vec<ChannelId>,
        /// Its replan rate (virtual mode).
        replan_hz: f64,
        /// One delivery contract per bound input stream.
        contracts: Vec<StreamContract>,
        /// One action grid per commanded part.
        grids: Vec<ActionGrid>,
    },
    /// A publisher installed the plan of a stream's demand revision.
    DemandAck {
        /// The stream.
        channel: ChannelId,
        /// The installed revision.
        revision: u64,
    },
    /// Request (or renew) leases on commandable parts, all or none.
    GrantLease {
        /// The parts.
        channels: Vec<ChannelId>,
        /// Parts the requester releases in the same revision (a switch
        /// between members of one command group).
        release: Vec<ChannelId>,
        /// The binding the leases are bound to: the requester's current one.
        binding: Fingerprint,
        /// `None` takes the session's lease grace.
        ttl_ns: Option<u64>,
    },
    /// Give leases back.
    ReleaseLease {
        /// The parts.
        channels: Vec<ChannelId>,
        /// The token held on each.
        token: LeaseToken,
    },
    /// Emergency stop: accepted from anyone.
    Estop {
        /// Why.
        reason: String,
        /// [`ReasonCode::EstopLocal`] or [`ReasonCode::EstopSignal`] when an
        /// edge reports its own latch; anything else is a remote e-stop.
        code: ReasonCode,
    },
    /// An embodiment's local clear of a latched e-stop.
    Clear,
    /// An embodiment fell back on its own account (its edge, or a device
    /// watchdog): ACTIVE, ARMED and DISARMED go SAFE; SAFE and ESTOP record
    /// why.
    EnterSafe {
        /// Why.
        reason: String,
        /// The cause.
        code: ReasonCode,
    },
    /// Virtual mode: an embodiment published tick `tick`'s samples.
    Ready {
        /// The tick.
        tick: u64,
        /// The last seq per stream it publishes.
        watermarks: Vec<Watermark>,
    },
    /// Virtual mode: a due controller's decision for tick `tick`.
    Decided {
        /// The tick.
        tick: u64,
        /// The chunks it sent for it (none: no action).
        chunks: Vec<ChunkRef>,
    },
    /// What a key delivery changed at the sender (§8.5).
    KeysAck(KeysAck),
    /// Every key issued to the sender's generation, in the reply (§8.5).
    GetKeys,
}

/// A verb from a participant.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The sender.
    pub from: ParticipantId,
    /// The epoch the sender believes current; lease grants and tick verbs
    /// enforce it.
    pub epoch: Option<Epoch>,
    /// Rising per participant generation from 1; a retry keeps it.
    pub request_id: u64,
    /// The verb.
    pub command: Command,
}

/// Why the host refused a control verb.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CtlStatus {
    /// Malformed or meaningless.
    InvalidRequest,
    /// The sender is not in the roster.
    NotAttached,
    /// The sender's roles do not allow it.
    Forbidden,
    /// Not allowed in the current mode.
    InvalidTransition,
    /// An e-stop is latched.
    EstopLatched,
    /// Issued against an older epoch.
    StaleEpoch,
    /// Carries a token that is not current.
    StaleToken,
    /// Held by an equal or higher class.
    LeaseBusy,
    /// No such (commandable) part.
    UnknownPath,
    /// Declared but not implemented (a merge policy).
    Unsupported,
    /// The sender requires a feature (or protocol) the session lacks.
    Incompatible,
    /// A member of the part's command group is held by an equal or higher
    /// class, or by the requester without releasing it.
    GroupBusy,
    /// A lease on a part outside the requester's binding.
    NotBound,
    /// A lease requested before every stream of the requester's binding
    /// installed a plan that includes its contract.
    NotReady,
    /// No valid handshake: unknown, superseded or forged keys.
    Unauthenticated,
    /// The sender's device may not do this.
    NotAuthorized,
    /// A request id below the host's reply window: never applied, and its
    /// first outcome is unknown.
    Expired,
}

/// A refused control verb.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{status:?}: {message}")]
pub struct CtlError {
    /// The refusal class.
    pub status: CtlStatus,
    /// A human explanation.
    pub message: String,
}

impl CtlError {
    /// A refusal.
    pub fn new(status: CtlStatus, message: impl fmt::Display) -> Self {
        Self {
            status,
            message: message.to_string(),
        }
    }
}

/// The host's answer to a verb.
#[derive(Clone, Debug, PartialEq)]
pub struct CtlReply {
    /// `Ok` or the refusal.
    pub result: Result<(), CtlError>,
    /// The authority snapshot right after the verb, accepted or not.
    pub state: AuthoritySnapshot,
    /// The leases granted, one per part (`GrantLease`).
    pub leases: Vec<Lease>,
    /// The request this answers.
    pub request_id: u64,
    /// Keys for the caller (a grant's, or `GetKeys`'), sealed with the
    /// reply.
    pub keys: Vec<IssuedKey>,
}

/// msg 1 of the handshake, on `hs`: Noise IK with an empty payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandshakeRequest {
    /// The participant id it attaches as.
    pub participant: ParticipantId,
    /// The Noise message.
    pub noise: Vec<u8>,
}

/// msg 2's payload: what the host issues a participant generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandshakeAccept {
    /// The generation, rising per host instance.
    pub generation: u32,
    /// The device the initiator's static key belongs to.
    pub device: String,
    /// The highest class the device may attach at (`None`: observer
    /// only).
    pub max_priority: Option<PriorityClass>,
    /// The host's Ed25519 key, which signs `host/state` and `host/tick`.
    pub host_sign_key: [u8; 32],
    /// The host instance, which those broadcasts carry.
    pub host_instance: u64,
    /// The id of `k_up` (participant to host).
    pub up_key_id: u32,
    /// The id of `k_down` (host to participant).
    pub down_key_id: u32,
}
