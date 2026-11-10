//! What the authority reports: the events its verbs cause and why a
//! participant left.

use core::fmt;

use crate::prelude::*;
use crate::wire::{
    ChannelId, Demand, Epoch, Fingerprint, Lease, Mode, ParticipantId, PriorityClass, Reason, Role,
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
    },
    /// A higher class took a held lease (on the part itself, or on another
    /// member of its command group).
    LeasePreempted {
        /// The fenced lease.
        previous: Lease,
        /// The new lease.
        by: Lease,
    },
    /// A rebinding re-issued a lease under a fresh token.
    LeaseRebound {
        /// The fenced lease.
        previous: Lease,
        /// The lease under the new binding.
        lease: Lease,
    },
    /// A lease ended.
    LeaseEnded {
        /// The lease.
        lease: Lease,
        /// Why.
        reason: Reason,
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
    },
    /// Parts left the table (removed, re-declared with another schema, or
    /// their owner detached); their ids are retired.
    PartsRemoved {
        /// Their robot.
        owner: ParticipantId,
        /// The retired stream ids.
        channels: Vec<ChannelId>,
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
        /// Its new demand.
        demand: Option<Demand>,
    },
    /// A publisher installed a stream's plan.
    DemandInstalled {
        /// The stream.
        channel: ChannelId,
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
            Self::LeaseGranted { lease } => write!(
                f,
                "lease {} granted to {} ({:?}) token {} binding {}",
                lease.channel, lease.holder, lease.priority, lease.token, lease.binding
            ),
            Self::LeasePreempted { previous, by } => write!(
                f,
                "lease {} of {} ({:?}, {}) preempted by {} on {} ({:?}, {})",
                previous.channel,
                previous.holder,
                previous.priority,
                previous.token,
                by.holder,
                by.channel,
                by.priority,
                by.token
            ),
            Self::LeaseRebound { previous, lease } => write!(
                f,
                "lease {} of {} rebound: {} -> {} (binding {})",
                lease.channel, lease.holder, previous.token, lease.token, lease.binding
            ),
            Self::LeaseEnded { lease, reason } => write!(
                f,
                "lease {} of {} token {} ended ({reason})",
                lease.channel, lease.holder, lease.token
            ),
            Self::EstopLatched { by, reason } => write!(f, "ESTOP latched by {by}: {reason}"),
            Self::EstopCleared { by } => write!(f, "ESTOP latch cleared by {by}"),
            Self::PartsDeclared { owner, channels } => {
                write!(f, "{owner} declared parts {channels:?}")
            }
            Self::PartsRemoved { owner, channels } => {
                write!(f, "parts {channels:?} of {owner} removed")
            }
            Self::BindingDeclared {
                participant,
                fingerprint,
            } => write!(f, "{participant} bound {fingerprint}"),
            Self::DemandChanged {
                channel,
                demand: Some(d),
            } => write!(f, "demand {channel} revision {}: {:?}", d.revision, d.plan),
            Self::DemandChanged {
                channel,
                demand: None,
            } => write!(f, "demand {channel} withdrawn"),
            Self::DemandInstalled { channel, revision } => {
                write!(f, "demand {channel} revision {revision} installed")
            }
            Self::BarrierTimeout { tick } => write!(f, "virtual barrier timed out at tick {tick}"),
        }
    }
}
