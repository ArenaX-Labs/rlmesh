//! Authority state and control verbs: roles, the session mode, leases, clock
//! mappings, the host's snapshot, and the `ctl` request/reply pair.

use std::fmt;

use crate::channel::ChannelSpec;
use crate::ids::{ChannelId, Epoch, LeaseToken, ParticipantId, SessionId};

/// What a participant does in a session. One participant may hold several.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// Publishes sensor channels and consumes actuation (real, sim or twin).
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
    /// The capabilities it requires.
    pub capabilities: Vec<String>,
}

impl ParticipantInfo {
    /// Whether it holds `role`.
    pub fn has_role(&self, role: Role) -> bool {
        self.roles.contains(&role)
    }
}

/// Command authority over one actuation channel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lease {
    /// The actuation channel.
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
    /// The channel has no lease.
    #[error("no lease on {channel}")]
    NoLease {
        /// The channel.
        channel: ChannelId,
    },
    /// Fenced: not the current lease's token (a preempted holder).
    #[error("token {got} on {channel} is fenced; the current lease is {current}")]
    StaleToken {
        /// The channel.
        channel: ChannelId,
        /// The current token.
        current: LeaseToken,
        /// The one it carried.
        got: LeaseToken,
    },
    /// The lease expired.
    #[error("the lease on {channel} expired")]
    Expired {
        /// The channel.
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
    /// The declared channel table.
    pub channels: Vec<ChannelSpec>,
    /// The host's protocol minor.
    pub protocol_minor: u32,
    /// The capabilities the session supports.
    pub capabilities: Vec<String>,
}

impl AuthoritySnapshot {
    /// A declared channel.
    pub fn channel(&self, id: ChannelId) -> Option<&ChannelSpec> {
        self.channels.iter().find(|c| c.id == id)
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

    /// The fence an edge applies to an incoming chunk: right epoch, and the
    /// token of the channel's current lease. The edge checks expiry itself,
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
        /// Capabilities it requires of the session (refused with
        /// [`CtlStatus::Incompatible`] when the host lacks one).
        capabilities: Vec<String>,
    },
    /// Leave, releasing leases and removing owned channels.
    Detach,
    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    Arm,
    /// Back to DISARMED, revoking every lease.
    Disarm,
    /// Request (or renew) the lease on an actuation channel.
    GrantLease {
        /// The channel.
        channel: ChannelId,
        /// `None` takes the session's lease grace.
        ttl_ns: Option<u64>,
    },
    /// Give a lease back.
    ReleaseLease {
        /// The channel.
        channel: ChannelId,
        /// The token held.
        token: LeaseToken,
    },
    /// Emergency stop: accepted from anyone.
    Estop {
        /// Why.
        reason: String,
    },
    /// An embodiment's local clear of a latched e-stop.
    Clear,
    /// Declares channels owned by the sender; an identical re-declaration is
    /// a no-op.
    DeclareChannels(Vec<ChannelSpec>),
    /// An embodiment's edge fell back on its own account: ACTIVE goes SAFE.
    EnterSafe {
        /// Why.
        reason: String,
    },
}

/// A verb from a participant.
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    /// The sender.
    pub from: ParticipantId,
    /// The epoch the sender believes current; lease grants enforce it.
    pub epoch: Option<Epoch>,
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
    /// The channel id is declared differently or by another owner.
    ChannelConflict,
    /// No such (actuation) channel.
    UnknownChannel,
    /// Declared but not implemented (a merge policy, an encoding).
    Unsupported,
    /// The sender requires a capability the session lacks.
    Incompatible,
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
    /// The granted lease, for `GrantLease`.
    pub lease: Option<Lease>,
}
