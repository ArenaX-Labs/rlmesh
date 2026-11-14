//! What the authority reports: the events its verbs cause and why a
//! participant left.

use core::fmt;

use crate::prelude::*;
use crate::wire::{
    ChannelId, Demand, Epoch, Fingerprint, Lease, Mode, ParticipantId, PriorityClass, Reason,
    ReasonCode, Role,
};

/// Why a participant left the roster.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DetachReason {
    /// It sent `Detach`.
    Requested,
    /// Its heartbeats lapsed (crash, network loss).
    Lost,
    /// Its device's entry changed: it re-attaches under the new one.
    AccessChanged,
    /// Its device was revoked.
    Revoked,
}

/// Something the authority state did, for publication, logging and evidence.
#[derive(Clone, Debug, PartialEq)]
pub enum AuthorityEvent {
    /// A participant joined (or re-joined) the roster.
    Attached {
        /// Who.
        participant: ParticipantId,
        /// Its roles.
        roles: Vec<Role>,
        /// Its lease class.
        priority: PriorityClass,
    },
    /// A participant left the roster.
    Detached {
        /// Who.
        participant: ParticipantId,
        /// Why.
        reason: DetachReason,
    },
    /// The session mode changed.
    ModeChanged {
        /// The old mode.
        from: Mode,
        /// The new mode.
        to: Mode,
        /// The epoch after the change.
        epoch: Epoch,
        /// Why.
        reason: Reason,
    },
    /// A stopped mode (SAFE or ESTOP) recorded a later cause without
    /// changing.
    ModeReason {
        /// The mode.
        mode: Mode,
        /// The cause.
        reason: Reason,
    },
    /// A lease was granted on a free part.
    LeaseGranted {
        /// The lease.
        lease: Lease,
        /// Its part, `robot:path`.
        part: String,
    },
    /// A higher class took a held lease (on the part itself, or on another
    /// member of its command group).
    LeasePreempted {
        /// The fenced lease.
        previous: Lease,
        /// Its part, `robot:path`.
        part: String,
        /// The new lease.
        by: Lease,
        /// Its part, `robot:path`.
        by_part: String,
    },
    /// A rebinding re-issued a lease under a fresh token.
    LeaseRebound {
        /// The fenced lease.
        previous: Lease,
        /// The lease under the new binding.
        lease: Lease,
        /// Its part, `robot:path`.
        part: String,
    },
    /// A lease ended.
    LeaseEnded {
        /// The lease.
        lease: Lease,
        /// Its part, `robot:path`.
        part: String,
        /// Why.
        reason: Reason,
    },
    /// A lease holder went away: the leases it held ended, and nothing else
    /// stopped for it.
    HolderLost {
        /// The holder.
        holder: ParticipantId,
        /// How it went (`HOLDER_LOST`, `HEARTBEAT_LAPSE`, `HOLDER_DETACHED`).
        code: ReasonCode,
        /// The parts released, each `robot:part`.
        parts: Vec<String>,
    },
    /// A robot stopped on its own: it takes no lease until re-armed by name.
    RobotHeld {
        /// The robot.
        robot: ParticipantId,
        /// Why.
        reason: Reason,
    },
    /// A robot's hold was cleared.
    RobotRearmed {
        /// The robot.
        robot: ParticipantId,
        /// Who re-armed it.
        by: ParticipantId,
    },
    /// An e-stop latched.
    EstopLatched {
        /// Who sent it.
        by: ParticipantId,
        /// Why.
        reason: String,
    },
    /// An embodiment cleared the latch.
    EstopCleared {
        /// The embodiment.
        by: ParticipantId,
    },
    /// Parts joined the table.
    PartsDeclared {
        /// Their robot.
        owner: ParticipantId,
        /// Their new stream ids.
        channels: Vec<ChannelId>,
        /// The parts, each `robot:path`.
        parts: Vec<String>,
    },
    /// Parts left the table (removed, re-declared with another schema, or
    /// their owner detached); their ids are retired.
    PartsRemoved {
        /// Their robot.
        owner: ParticipantId,
        /// The retired stream ids.
        channels: Vec<ChannelId>,
        /// The parts, each `robot:path`.
        parts: Vec<String>,
    },
    /// A controller declared a new binding.
    BindingDeclared {
        /// The controller.
        participant: ParticipantId,
        /// Its fingerprint.
        fingerprint: Fingerprint,
    },
    /// A stream's publication plan changed (or went away: `None`).
    DemandChanged {
        /// The stream.
        channel: ChannelId,
        /// Its part, `robot:path`.
        part: String,
        /// Its new demand.
        demand: Option<Demand>,
    },
    /// A publisher installed a stream's plan.
    DemandInstalled {
        /// The stream.
        channel: ChannelId,
        /// Its part, `robot:path`.
        part: String,
        /// The installed revision.
        revision: u64,
    },
    /// The virtual-mode barrier waited past its timeout.
    BarrierTimeout {
        /// The tick it was on.
        tick: u64,
    },
}

impl fmt::Display for AuthorityEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Attached {
                participant,
                roles,
                priority,
            } => write!(f, "attached {participant} {roles:?} as {priority:?}"),
            Self::Detached {
                participant,
                reason,
            } => write!(f, "detached {participant} ({reason:?})"),
            Self::ModeChanged {
                from,
                to,
                epoch,
                reason,
            } => write!(f, "mode {from} -> {to} at {epoch} ({reason})"),
            Self::ModeReason { mode, reason } => write!(f, "mode {mode} stays ({reason})"),
            Self::LeaseGranted { lease, part } => write!(
                f,
                "lease {part} granted to {} ({:?}) token {} binding {}",
                lease.holder, lease.priority, lease.token, lease.binding
            ),
            Self::LeasePreempted {
                previous,
                part,
                by,
                by_part,
            } => write!(
                f,
                "lease {part} of {} ({:?}, {}) preempted by {} on {by_part} ({:?}, {})",
                previous.holder,
                previous.priority,
                previous.token,
                by.holder,
                by.priority,
                by.token
            ),
            Self::LeaseRebound {
                previous,
                lease,
                part,
            } => write!(
                f,
                "lease {part} of {} rebound: {} -> {} (binding {})",
                lease.holder, previous.token, lease.token, lease.binding
            ),
            Self::LeaseEnded {
                lease,
                part,
                reason,
            } => write!(
                f,
                "lease {part} of {} token {} ended ({reason})",
                lease.holder, lease.token
            ),
            Self::HolderLost {
                holder,
                code,
                parts,
            } => write!(
                f,
                "holder {holder} lost ({code}): released {}",
                parts.join(", ")
            ),
            Self::RobotHeld { robot, reason } => write!(f, "robot {robot} held ({reason})"),
            Self::RobotRearmed { robot, by } => write!(f, "robot {robot} re-armed by {by}"),
            Self::EstopLatched { by, reason } => write!(f, "ESTOP latched by {by}: {reason}"),
            Self::EstopCleared { by } => write!(f, "ESTOP latch cleared by {by}"),
            Self::PartsDeclared { owner, parts, .. } => {
                write!(f, "{owner} declared parts {}", parts.join(", "))
            }
            Self::PartsRemoved { owner, parts, .. } => {
                write!(f, "parts {} of {owner} removed", parts.join(", "))
            }
            Self::BindingDeclared {
                participant,
                fingerprint,
            } => write!(f, "{participant} bound {fingerprint}"),
            Self::DemandChanged {
                part,
                demand: Some(d),
                ..
            } => write!(f, "demand {part} revision {}: {:?}", d.revision, d.plan),
            Self::DemandChanged {
                part, demand: None, ..
            } => write!(f, "demand {part} withdrawn"),
            Self::DemandInstalled { part, revision, .. } => {
                write!(f, "demand {part} revision {revision} installed")
            }
            Self::BarrierTimeout { tick } => write!(f, "virtual barrier timed out at tick {tick}"),
        }
    }
}
