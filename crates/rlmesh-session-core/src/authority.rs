//! Authority state (§9.7, §10.5, §11): epoch, roster, mode, e-stop latch,
//! the part table, bindings and the lease table, with the session host as
//! its single writer.
//!
//! [`Authority`] is sans-IO: every method takes the host's session clock
//! (`now_ns`) and returns what it caused, so the state machine is tested
//! without a network and the host only has to serialise calls into it.
//!
//! - **Parts** come from an embodiment's authored map, normalised by the
//!   host with the shared normaliser ([`crate::spec::normalise()`]); the
//!   snapshot carries each robot's declaration fingerprint. Channel ids are
//!   host-assigned and never reused within the session: re-declaring a path
//!   with a changed schema retires its id and assigns a new one. A
//!   commandable part is not re-declared while ARMED or ACTIVE
//!   (`INVALID_TRANSITION`); sensor parts can be added at any time.
//! - **Leases** are per commandable part, granted all or none
//!   ([`Command::GrantLease`]) and bound to the holder's current
//!   **binding**: chunks must carry its fingerprint, and a rebinding
//!   re-issues the holder's leases under a fresh token, so in-flight chunks
//!   of the old binding are fenced (Codex finding 2).
//! - **Delivery contracts** (§8.2) come with a binding, one per bound input
//!   stream. The host admits them ([`crate::spec::contract::admit`]), derives
//!   one publication plan per stream from every attached controller's
//!   contracts and publishes it as a [`Demand`]; the stream's publisher
//!   installs it and acknowledges the revision ([`Command::DemandAck`]).
//!   Install before authority: a lease is granted only once every stream of
//!   the requester's binding has acknowledged a revision that includes its
//!   contract (`NOT_READY`). Demand is recomputed on attach, detach,
//!   participant loss, part removal and rebinding.
//! - Priority classes: `safety > operator > model > scripted`. A higher
//!   class preempts a lower one and gets a fresh token; an equal or lower
//!   class gets `LEASE_BUSY`.
//! - **Command groups** are scoped `(robot, group)`: at most one member is
//!   leased at a time. A member held by an equal or higher class, or by the
//!   requester without releasing it in the same request, is `GROUP_BUSY`; a
//!   lower class's member is preempted. `release` switches members
//!   atomically (Codex finding 3).
//! - Leases have a TTL, renewed by every heartbeat from the holder on the
//!   current epoch; a lapse while ACTIVE takes the session SAFE.
//! - ESTOP is accepted from anyone (refusing an e-stop is never safer) and
//!   latches; only an embodiment's local clear releases the latch, and then a
//!   disarm and an arm (a fresh epoch) are needed.
//! - **Reasons** (DESIGN §7): every mode change records a structured
//!   [`Reason`] (`mode_reason`), every lease ending goes into a bounded ring
//!   of [`LeaseEnd`]s, and `last_stop` marks the last entry into SAFE,
//!   DISARMED or ESTOP, so clients never infer why. An embodiment's
//!   `EnterSafe` takes ACTIVE, ARMED and DISARMED to SAFE; SAFE and ESTOP
//!   stay and record its reason.
//! - In **virtual** clock mode the authority also runs the tick barrier
//!   ([`crate::virtual_clock`]); the ticks it emits are drained with
//!   [`Authority::take_ticks`].

#[allow(unused_imports)]
use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use core::fmt;

use crate::spec::contract;
use crate::wire::authority::fence;
use crate::wire::part::PartSpec;
use crate::wire::tick::VirtualTick;
use crate::wire::version::{PROTOCOL_MINOR, missing_features};
use crate::wire::{
    AuthoritySnapshot, BindingRecord, ChannelId, ClockMapping, ClockMode, Command, CtlError,
    CtlStatus, Demand, Epoch, FenceError, Fingerprint, LastStop, Lease, LeaseEnd, LeaseToken, Mode,
    ParticipantId, ParticipantInfo, PriorityClass, Reason, ReasonCode, Request, Role, SessionId,
};

use crate::virtual_clock::{Participants, VirtualClock, VirtualConfig};

/// The highest replan rate a binding declares.
const MAX_REPLAN_HZ: f64 = 10_000.0;
/// The longest action period a binding declares.
const MAX_ACTION_DT: core::time::Duration = core::time::Duration::from_secs(60);
/// Lease endings the snapshot keeps.
const ENDED: usize = 16;

/// Why a participant left the roster.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DetachReason {
    /// It sent `Detach`.
    Requested,
    /// Its liveliness token went away (crash, network loss).
    LivelinessLost,
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

/// The authority's rules.
#[derive(Clone, Debug, PartialEq)]
pub struct AuthorityConfig {
    /// Default lease ttl: how long the holder's heartbeats may stop before
    /// its lease lapses.
    pub lease_grace_ns: u64,
    /// The features the session implements (see [`crate::wire::version`]).
    pub supports: Vec<String>,
    /// Virtual clock mode (§11.3); `None` is wall mode.
    pub virtual_clock: Option<VirtualConfig>,
}

impl Default for AuthorityConfig {
    fn default() -> Self {
        Self {
            lease_grace_ns: 250_000_000,
            supports: crate::wire::version::FEATURES
                .iter()
                .map(ToString::to_string)
                .collect(),
            virtual_clock: None,
        }
    }
}

/// What an accepted verb produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Applied {
    /// The leases granted, for `GrantLease`.
    pub leases: Vec<Lease>,
    /// What changed.
    pub events: Vec<AuthorityEvent>,
}

/// The authority state machine. Only the session host owns one.
#[derive(Clone, Debug)]
pub struct Authority {
    session: SessionId,
    config: AuthorityConfig,
    epoch: Epoch,
    revision: u64,
    mode: Mode,
    estop_latched: bool,
    mode_reason: Reason,
    /// The latest lease endings, oldest first.
    ended: VecDeque<LeaseEnd>,
    last_stop: LastStop,
    /// The host clock of the call in progress, for the reasons it records.
    now_ns: i64,
    roster: BTreeMap<ParticipantId, ParticipantInfo>,
    leases: BTreeMap<ChannelId, Lease>,
    /// Declared parts by stream id.
    parts: BTreeMap<ChannelId, PartSpec>,
    /// The last channel id assigned; ids only grow.
    last_channel: u32,
    bindings: BTreeMap<ParticipantId, BindingRecord>,
    /// Each robot's declaration fingerprint (§3.3).
    contract: BTreeMap<ParticipantId, Fingerprint>,
    /// The publication plan per stream some attached binding contracts.
    demands: BTreeMap<ChannelId, Demand>,
    /// The last demand revision per stream, kept when a demand goes away so
    /// a stale acknowledgement never satisfies a later one.
    demand_revisions: BTreeMap<ChannelId, u64>,
    /// Per controller and stream, the demand revision that first included
    /// its contract: what its publisher must have acknowledged.
    needed: BTreeMap<(ParticipantId, ChannelId), u64>,
    last_token: u64,
    clock: Option<VirtualClock>,
    /// Virtual ticks to publish.
    ticks: Vec<VirtualTick>,
}

impl Authority {
    /// A fresh session: DISARMED in epoch 1.
    pub fn new(session: SessionId, config: AuthorityConfig) -> Self {
        // Epoch 0 means "unchecked" on the wire, so sessions start at 1.
        let clock = config
            .virtual_clock
            .map(|virtual_clock| VirtualClock::new(virtual_clock, Epoch::FIRST, 0));
        Self {
            session,
            config,
            epoch: Epoch::FIRST,
            revision: 0,
            mode: Mode::Disarmed,
            estop_latched: false,
            mode_reason: Reason::default(),
            ended: VecDeque::with_capacity(ENDED),
            // A fresh session is stopped (DISARMED) in its first epoch.
            last_stop: LastStop {
                epoch: Epoch::FIRST,
                revision: 0,
            },
            now_ns: 0,
            roster: BTreeMap::new(),
            leases: BTreeMap::new(),
            parts: BTreeMap::new(),
            last_channel: 0,
            bindings: BTreeMap::new(),
            contract: BTreeMap::new(),
            demands: BTreeMap::new(),
            demand_revisions: BTreeMap::new(),
            needed: BTreeMap::new(),
            last_token: 0,
            clock,
            ticks: Vec::new(),
        }
    }

    /// The session.
    pub fn session(&self) -> SessionId {
        self.session
    }

    /// Its rules.
    pub fn config(&self) -> &AuthorityConfig {
        &self.config
    }

    /// The current epoch.
    pub fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// The structural revision.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Whether an e-stop is latched.
    pub fn estop_latched(&self) -> bool {
        self.estop_latched
    }

    /// Why the mode is what it is.
    pub fn mode_reason(&self) -> &Reason {
        &self.mode_reason
    }

    /// The last entry into SAFE, DISARMED or ESTOP.
    pub fn last_stop(&self) -> LastStop {
        self.last_stop
    }

    /// The lease on `channel`.
    pub fn lease(&self, channel: ChannelId) -> Option<&Lease> {
        self.leases.get(&channel)
    }

    /// A declared part, by its stream id.
    pub fn part(&self, channel: ChannelId) -> Option<&PartSpec> {
        self.parts.get(&channel)
    }

    /// Every declared part.
    pub fn parts(&self) -> impl Iterator<Item = &PartSpec> {
        self.parts.values()
    }

    /// A controller's binding.
    pub fn binding(&self, id: &ParticipantId) -> Option<&BindingRecord> {
        self.bindings.get(id)
    }

    /// Leases held.
    pub fn lease_count(&self) -> usize {
        self.leases.len()
    }

    /// Whether `id` is in the roster.
    pub fn is_attached(&self, id: &ParticipantId) -> bool {
        self.roster.contains_key(id)
    }

    /// The virtual ticks emitted since the last call, in order.
    pub fn take_ticks(&mut self) -> Vec<VirtualTick> {
        core::mem::take(&mut self.ticks)
    }

    /// The host-side fence, expiry included.
    pub fn check_fence(
        &self,
        channel: ChannelId,
        token: LeaseToken,
        epoch: Epoch,
        now_ns: i64,
    ) -> Result<(), FenceError> {
        let lease = fence(self.epoch, self.leases.get(&channel), channel, token, epoch)?;
        if now_ns > lease.expires_ns {
            return Err(FenceError::Expired { channel });
        }
        Ok(())
    }

    /// The snapshot at `now_ns`, with the host's clock mappings.
    pub fn snapshot(&self, clocks: Vec<ClockMapping>, now_ns: i64) -> AuthoritySnapshot {
        AuthoritySnapshot {
            session: self.session,
            epoch: self.epoch,
            revision: self.revision,
            mode: self.mode,
            estop_latched: self.estop_latched,
            roster: self.roster.values().cloned().collect(),
            leases: self.leases.values().cloned().collect(),
            clocks,
            host_ns: now_ns,
            parts: self.parts.values().cloned().collect(),
            protocol_minor: PROTOCOL_MINOR,
            supports: self.config.supports.clone(),
            clock_mode: if self.clock.is_some() {
                ClockMode::Virtual
            } else {
                ClockMode::Wall
            },
            tick_ns: self.config.virtual_clock.map_or(0, |c| c.tick_ns),
            bindings: self.bindings.values().cloned().collect(),
            contract: self.contract.clone(),
            demands: self.demands.values().cloned().collect(),
            mode_reason: self.mode_reason.clone(),
            ended: self.ended.iter().cloned().collect(),
            last_stop: self.last_stop,
        }
    }

    /// Applies a verb. Accepted verbs that change state bump the revision.
    pub fn apply(&mut self, request: Request, now_ns: i64) -> Result<Applied, CtlError> {
        self.now_ns = now_ns;
        let mut applied = Applied::default();
        let from = request.from;
        match request.command {
            Command::Attach {
                name,
                roles,
                priority,
                protocol_minor,
                supports,
                requires,
                workflow_edition,
                workflow_editions,
            } => {
                if roles.is_empty() {
                    return Err(CtlError::new(
                        CtlStatus::InvalidRequest,
                        "attach declares no role",
                    ));
                }
                let missing = missing_features(&requires, &self.config.supports);
                if !missing.is_empty() {
                    return Err(CtlError::new(
                        CtlStatus::Incompatible,
                        format!("this session (protocol minor {PROTOCOL_MINOR}) lacks {missing:?}"),
                    ));
                }
                // A re-attach (reconnect) keeps the participant's leases.
                self.roster.insert(
                    from.clone(),
                    ParticipantInfo {
                        id: from.clone(),
                        name,
                        roles: roles.clone(),
                        priority,
                        attached_ns: now_ns,
                        protocol_minor,
                        supports,
                        requires,
                        workflow_edition,
                        workflow_editions,
                    },
                );
                applied.events.push(AuthorityEvent::Attached {
                    participant: from,
                    roles,
                    priority,
                });
                self.recompute_demands(&mut applied.events);
            }
            Command::Detach => {
                self.require_attached(&from)?;
                let why = self.reason(ReasonCode::HolderDetached, &from, "");
                self.end_leases_of(&from, &why, &mut applied.events);
                let owned: Vec<ChannelId> = self
                    .parts
                    .values()
                    .filter(|p| p.owner == from)
                    .map(|p| p.channel)
                    .collect();
                self.remove_parts(&from, &owned, &mut applied.events);
                self.contract.remove(&from);
                self.bindings.remove(&from);
                self.roster.remove(&from);
                self.needed.retain(|(p, _), _| p != &from);
                self.recompute_demands(&mut applied.events);
                applied.events.push(AuthorityEvent::Detached {
                    participant: from.clone(),
                    reason: DetachReason::Requested,
                });
                self.settle_after_release(&from, "holder detached", &mut applied.events);
                self.barrier_gone(&from, now_ns);
            }
            Command::Arm => {
                self.require_attached(&from)?;
                match self.mode {
                    Mode::Disarmed | Mode::Safe => {
                        self.epoch = self.epoch.next();
                        if let Some(clock) = &mut self.clock {
                            clock.reset(self.epoch, now_ns);
                        }
                        let why = self.reason(ReasonCode::Armed, &from, "");
                        self.set_mode(Mode::Armed, why, &mut applied.events);
                    }
                    Mode::Estop if self.estop_latched => {
                        return Err(CtlError::new(
                            CtlStatus::EstopLatched,
                            "e-stop is latched: clear it locally, then disarm and arm",
                        ));
                    }
                    Mode::Estop => {
                        return Err(CtlError::new(
                            CtlStatus::InvalidTransition,
                            "leaving ESTOP needs a disarm before arm",
                        ));
                    }
                    Mode::Armed | Mode::Active => {
                        return Err(CtlError::new(
                            CtlStatus::InvalidTransition,
                            format!("already {}", self.mode),
                        ));
                    }
                }
            }
            Command::Disarm => {
                self.require_attached(&from)?;
                match self.mode {
                    Mode::Disarmed => {}
                    Mode::Estop if self.estop_latched => {
                        return Err(CtlError::new(
                            CtlStatus::EstopLatched,
                            "e-stop is latched: clear it locally first",
                        ));
                    }
                    _ => {
                        let why = self.reason(ReasonCode::Disarmed, &from, "");
                        self.revoke_all(ReasonCode::VoidedByDisarm, &why, &mut applied.events);
                        self.set_mode(Mode::Disarmed, why, &mut applied.events);
                    }
                }
            }
            Command::DeclareParts {
                spec_version,
                robot_json,
            } => self.declare_parts(&from, spec_version, &robot_json, &mut applied.events)?,
            Command::DeclareBinding {
                fingerprint,
                revision,
                table_json,
                channels,
                replan_hz,
                contracts,
                grids,
            } => self.declare_binding(
                BindingRecord {
                    participant: from,
                    fingerprint,
                    revision,
                    table_json,
                    channels,
                    replan_hz,
                    contracts,
                    grids,
                },
                &mut applied.events,
            )?,
            Command::DemandAck { channel, revision } => {
                self.demand_ack(&from, channel, revision, &mut applied.events)?;
            }
            Command::GrantLease {
                channels,
                release,
                binding,
                ttl_ns,
            } => {
                applied.leases = self.grant(
                    &from,
                    request.epoch,
                    &Grant {
                        channels,
                        release,
                        binding,
                        ttl_ns,
                    },
                    now_ns,
                    &mut applied.events,
                )?;
            }
            Command::ReleaseLease { channels, token } => {
                self.release(&from, &channels, token, &mut applied.events)?;
            }
            Command::Estop { reason, code } => {
                // Accepted from anyone: refusing an e-stop is never safer.
                // TODO(auth): restrict to authorised participants and log
                // unauthorised attempts as evidence.
                if self.mode != Mode::Estop || !self.estop_latched {
                    // An edge reports its own latch; any other e-stop came
                    // through the host. Not verified until senders are
                    // authenticated.
                    let code = match code {
                        ReasonCode::EstopLocal | ReasonCode::EstopSignal => code,
                        _ => ReasonCode::EstopRemote,
                    };
                    let why = self.reason(code, &from, &reason);
                    self.revoke_all(ReasonCode::VoidedByEstop, &why, &mut applied.events);
                    self.estop_latched = true;
                    applied.events.push(AuthorityEvent::EstopLatched {
                        by: from.clone(),
                        reason,
                    });
                    if self.mode == Mode::Estop {
                        // Cleared but not yet disarmed: latched again.
                        self.mode_reason = why;
                    } else {
                        self.set_mode(Mode::Estop, why, &mut applied.events);
                    }
                }
            }
            Command::Clear => {
                let info = self.require_attached(&from)?;
                // "Local clear": only an embodiment may release its latch.
                if !info.has_role(Role::Embodiment) {
                    return Err(CtlError::new(
                        CtlStatus::Forbidden,
                        "only an embodiment can clear an e-stop",
                    ));
                }
                if !(self.mode == Mode::Estop && self.estop_latched) {
                    return Err(CtlError::new(
                        CtlStatus::InvalidTransition,
                        "no latched e-stop to clear",
                    ));
                }
                self.estop_latched = false;
                self.mode_reason = self.reason(ReasonCode::EstopCleared, &from, "");
                applied
                    .events
                    .push(AuthorityEvent::EstopCleared { by: from });
            }
            Command::EnterSafe { reason, code } => {
                let info = self.require_attached(&from)?;
                if !info.has_role(Role::Embodiment) {
                    return Err(CtlError::new(
                        CtlStatus::Forbidden,
                        "only an embodiment's edge reports a fallback",
                    ));
                }
                let why = self.reason(code, &from, &reason);
                match self.mode {
                    // A device fault between episodes is visible at once,
                    // and the next drive needs an arm.
                    Mode::Active | Mode::Armed | Mode::Disarmed => {
                        self.revoke_all(ReasonCode::VoidedBySafe, &why, &mut applied.events);
                        self.set_mode(Mode::Safe, why, &mut applied.events);
                    }
                    Mode::Safe | Mode::Estop => {
                        self.mode_reason = why.clone();
                        applied.events.push(AuthorityEvent::ModeReason {
                            mode: self.mode,
                            reason: why,
                        });
                    }
                }
            }
            Command::Ready { tick, watermarks } => {
                self.require_tick_verb(&from, request.epoch)?;
                let who = self.barrier_participants();
                if let Some(clock) = &mut self.clock {
                    let ticks = clock.on_ready(&from, tick, &watermarks, &who, now_ns)?;
                    self.ticks.extend(ticks);
                }
            }
            Command::Decided { tick, chunks } => {
                self.require_tick_verb(&from, request.epoch)?;
                let who = self.barrier_participants();
                if let Some(clock) = &mut self.clock {
                    let ticks = clock.on_decided(&from, tick, &chunks, &who, now_ns)?;
                    self.ticks.extend(ticks);
                }
            }
        }
        if !applied.events.is_empty() {
            self.revision += 1;
        }
        Ok(applied)
    }

    /// A heartbeat from `from` arrived: renew its leases when it is on the
    /// current epoch. Renewal is not a structural change.
    pub fn heartbeat(&mut self, from: &ParticipantId, epoch: Epoch, now_ns: i64) {
        if epoch != self.epoch {
            return;
        }
        for lease in self.leases.values_mut().filter(|l| &l.holder == from) {
            lease.expires_ns = now_ns.saturating_add_unsigned(lease.ttl_ns);
        }
    }

    /// Lapses leases whose holder stopped heartbeating (ACTIVE goes SAFE),
    /// and ends a virtual run whose barrier stalled.
    pub fn expire(&mut self, now_ns: i64) -> Vec<AuthorityEvent> {
        self.now_ns = now_ns;
        let mut events = Vec::new();
        let lapsed: Vec<ChannelId> = self
            .leases
            .values()
            .filter(|lease| now_ns > lease.expires_ns)
            .map(|lease| lease.channel)
            .collect();
        let mut first = None;
        for channel in &lapsed {
            if let Some(lease) = self.leases.remove(channel) {
                let why = self.reason(ReasonCode::HeartbeatLapse, &lease.holder, "");
                first.get_or_insert_with(|| why.clone());
                self.end(lease, why, &mut events);
            }
        }
        if let Some(why) = first {
            self.enter_safe(why, &mut events);
        }
        if let Some(tick) = self.clock.as_mut().and_then(|c| c.check_timeout(now_ns)) {
            events.push(AuthorityEvent::BarrierTimeout { tick });
            let why = self.reason(
                ReasonCode::Unspecified,
                "",
                &format!("virtual barrier timed out at tick {tick}"),
            );
            self.enter_safe(why, &mut events);
        }
        if !events.is_empty() {
            self.revision += 1;
        }
        events
    }

    /// The participant's liveliness token went away: it is gone, and any
    /// lease it held lapses. Its parts and binding stay for a reconnect.
    pub fn participant_lost(&mut self, id: &ParticipantId, now_ns: i64) -> Vec<AuthorityEvent> {
        self.now_ns = now_ns;
        let mut events = Vec::new();
        if self.roster.remove(id).is_none() {
            return events;
        }
        let before = events.len();
        let why = self.reason(ReasonCode::HolderLost, id, "");
        self.end_leases_of(id, &why, &mut events);
        let lapsed = events.len() > before;
        events.push(AuthorityEvent::Detached {
            participant: id.clone(),
            reason: DetachReason::LivelinessLost,
        });
        if lapsed {
            self.enter_safe(why, &mut events);
        }
        self.needed.retain(|(p, _), _| p != id);
        self.recompute_demands(&mut events);
        self.barrier_gone(id, now_ns);
        self.revision += 1;
        events
    }

    fn require_attached(&self, id: &ParticipantId) -> Result<&ParticipantInfo, CtlError> {
        self.roster
            .get(id)
            .ok_or_else(|| CtlError::new(CtlStatus::NotAttached, format!("{id} is not attached")))
    }

    fn require_role(&self, id: &ParticipantId, role: Role) -> Result<&ParticipantInfo, CtlError> {
        let info = self.require_attached(id)?;
        if info.has_role(role) {
            Ok(info)
        } else {
            Err(CtlError::new(
                CtlStatus::Forbidden,
                format!("{id} is not a {role:?}"),
            ))
        }
    }

    fn next_token(&mut self) -> LeaseToken {
        self.last_token += 1;
        LeaseToken(self.last_token)
    }

    fn next_channel(&mut self) -> Result<ChannelId, CtlError> {
        self.last_channel = self.last_channel.checked_add(1).ok_or_else(|| {
            CtlError::new(
                CtlStatus::Unsupported,
                "this session has used every channel id",
            )
        })?;
        Ok(ChannelId(self.last_channel))
    }

    /// The robot map of `from`, normalised and diffed against its current
    /// parts: unchanged parts keep their ids, changed or new ones get fresh
    /// ids, and missing ones are removed. All or nothing.
    fn declare_parts(
        &mut self,
        from: &ParticipantId,
        spec_version: u32,
        robot_json: &str,
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<(), CtlError> {
        self.require_role(from, Role::Embodiment)?;
        let declared = crate::spec::normalise(from, spec_version, robot_json)?;
        let fingerprint = crate::spec::normalise::fingerprint(&declared);
        let current: BTreeMap<&str, &PartSpec> = self
            .parts
            .values()
            .filter(|p| &p.owner == from)
            .map(|p| (p.path.as_str(), p))
            .collect();
        let armed = matches!(self.mode, Mode::Armed | Mode::Active);
        let refuse_armed = |path: &str| {
            CtlError::new(
                CtlStatus::InvalidTransition,
                format!(
                    "{path}: a commandable part is not re-declared while {}; disarm first",
                    self.mode
                ),
            )
        };
        let mut keep = BTreeSet::new();
        let mut fresh = Vec::new();
        for mut spec in declared {
            match current.get(spec.path.as_str()) {
                Some(old) => {
                    spec.channel = old.channel;
                    spec.applied_channel = old.applied_channel;
                    if spec == **old {
                        keep.insert(old.channel);
                        continue;
                    }
                    if armed && (old.command.is_some() || spec.command.is_some()) {
                        return Err(refuse_armed(spec.path.as_str()));
                    }
                    fresh.push(spec);
                }
                None => {
                    if armed && spec.command.is_some() {
                        return Err(refuse_armed(spec.path.as_str()));
                    }
                    fresh.push(spec);
                }
            }
        }
        let removed: Vec<ChannelId> = current
            .values()
            .filter(|p| !keep.contains(&p.channel))
            .map(|p| {
                if armed && p.command.is_some() && !fresh.iter().any(|f| f.path == p.path) {
                    Err(refuse_armed(p.path.as_str()))
                } else {
                    Ok(p.channel)
                }
            })
            .collect::<Result<_, _>>()?;
        // Validated: mutate.
        let mut added = Vec::with_capacity(fresh.len());
        for mut spec in fresh {
            spec.channel = self.next_channel()?;
            spec.applied_channel = match spec.command {
                Some(_) => Some(self.next_channel()?),
                None => None,
            };
            added.push(spec);
        }
        self.remove_parts(from, &removed, events);
        self.contract.insert(from.clone(), fingerprint);
        if !added.is_empty() {
            events.push(AuthorityEvent::PartsDeclared {
                owner: from.clone(),
                channels: added.iter().map(|p| p.channel).collect(),
            });
            for spec in added {
                self.parts.insert(spec.channel, spec);
            }
        }
        Ok(())
    }

    /// Retires parts (their ids are never reused) and ends their leases.
    fn remove_parts(
        &mut self,
        owner: &ParticipantId,
        channels: &[ChannelId],
        events: &mut Vec<AuthorityEvent>,
    ) {
        if channels.is_empty() {
            return;
        }
        for id in channels {
            if let Some(part) = self.parts.remove(id) {
                for stream in [Some(part.channel), part.applied_channel]
                    .into_iter()
                    .flatten()
                {
                    if self.demands.remove(&stream).is_some() {
                        events.push(AuthorityEvent::DemandChanged {
                            channel: stream,
                            demand: None,
                        });
                    }
                }
            }
            if let Some(lease) = self.leases.remove(id) {
                let why = self.reason(ReasonCode::PartRemoved, owner, "");
                self.end(lease, why, events);
            }
        }
        events.push(AuthorityEvent::PartsRemoved {
            owner: owner.clone(),
            channels: channels.to_vec(),
        });
    }

    fn commandable(&self, channel: ChannelId) -> Option<&PartSpec> {
        self.parts.get(&channel).filter(|p| p.command.is_some())
    }

    /// The `(robot, group)` of a commandable part.
    fn group_of(&self, channel: ChannelId) -> Option<(&ParticipantId, &str)> {
        let part = self.parts.get(&channel)?;
        let group = part.command.as_ref()?.group.as_deref()?;
        Some((&part.owner, group))
    }

    /// The other members of `channel`'s command group.
    fn group_peers(&self, channel: ChannelId) -> Vec<ChannelId> {
        let Some(group) = self.group_of(channel) else {
            return Vec::new();
        };
        self.parts
            .keys()
            .copied()
            .filter(|c| *c != channel && self.group_of(*c) == Some(group))
            .collect()
    }

    fn declare_binding(
        &mut self,
        record: BindingRecord,
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<(), CtlError> {
        let from = record.participant.clone();
        self.require_role(&from, Role::Controller)?;
        let unique: BTreeSet<ChannelId> = record.channels.iter().copied().collect();
        if unique.len() != record.channels.len() {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                "a binding lists a part twice",
            ));
        }
        if let Some(missing) = record
            .channels
            .iter()
            .find(|c| self.commandable(**c).is_none())
        {
            return Err(CtlError::new(
                CtlStatus::UnknownPath,
                format!("{missing} is not a commandable part"),
            ));
        }
        if !(record.replan_hz.is_finite() && (0.0..=MAX_REPLAN_HZ).contains(&record.replan_hz)) {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                format!("replan rate {}", record.replan_hz),
            ));
        }
        self.admit_contracts(&record)?;
        if let Some(current) = self.bindings.get(&from)
            && current.fingerprint == record.fingerprint
        {
            let same: BTreeSet<ChannelId> = current.channels.iter().copied().collect();
            if same != unique {
                return Err(CtlError::new(
                    CtlStatus::InvalidRequest,
                    format!(
                        "fingerprint {} already names a binding over other parts",
                        record.fingerprint
                    ),
                ));
            }
            return Ok(());
        }
        // A new binding: the holder's leases are re-issued under a fresh
        // token (or released when the new binding leaves their part out), so
        // every in-flight chunk of the old binding is fenced.
        let held: Vec<ChannelId> = self
            .leases
            .values()
            .filter(|l| l.holder == from)
            .map(|l| l.channel)
            .collect();
        let token = (!held.is_empty()).then(|| self.next_token());
        for channel in held {
            let Some(previous) = self.leases.remove(&channel) else {
                continue;
            };
            match token.filter(|_| unique.contains(&channel)) {
                Some(token) => {
                    let lease = Lease {
                        token,
                        binding: record.fingerprint,
                        ..previous.clone()
                    };
                    self.leases.insert(channel, lease.clone());
                    let why = self.reason(ReasonCode::Rebound, &from, "");
                    self.record_end(&previous, why, Some(&lease));
                    events.push(AuthorityEvent::LeaseRebound { previous, lease });
                }
                None => {
                    let why = self.reason(ReasonCode::Released, &from, "rebound without the part");
                    self.end(previous, why, events);
                }
            }
        }
        events.push(AuthorityEvent::BindingDeclared {
            participant: from.clone(),
            fingerprint: record.fingerprint,
        });
        self.needed.retain(|(p, _), _| p != &from);
        self.bindings.insert(from.clone(), record);
        self.recompute_demands(events);
        self.settle_after_release(&from, "rebound without its parts", events);
        Ok(())
    }

    /// The part a stream belongs to, and whether it is the part's applied
    /// stream.
    fn stream_part(&self, channel: ChannelId) -> Option<(&PartSpec, bool)> {
        self.parts.values().find_map(|p| {
            if p.channel == channel {
                Some((p, false))
            } else if p.applied_channel == Some(channel) {
                Some((p, true))
            } else {
                None
            }
        })
    }

    /// Admission (§8.2): every contract names a stream once and is one the
    /// stream can meet, the rings fit the cap, and every grid is on a
    /// commanded part.
    fn admit_contracts(&self, record: &BindingRecord) -> Result<(), CtlError> {
        let invalid = |message: String| CtlError::new(CtlStatus::InvalidRequest, message);
        let mut seen = BTreeSet::new();
        let mut rings = Vec::with_capacity(record.contracts.len());
        for c in &record.contracts {
            if !seen.insert(c.channel) {
                return Err(invalid(format!("two contracts on {}", c.channel)));
            }
            let (part, applied) = self.stream_part(c.channel).ok_or_else(|| {
                CtlError::new(
                    CtlStatus::UnknownPath,
                    format!("{} is not a declared stream", c.channel),
                )
            })?;
            contract::admit(c, part, applied).map_err(invalid)?;
            rings.push((c, part, applied));
        }
        let bytes = contract::ring_bytes(rings);
        if bytes > contract::MAX_RING_BYTES {
            return Err(invalid(format!(
                "the binding's windows hold {} MiB (at most 512 MiB)",
                bytes >> 20
            )));
        }
        let mut gridded = BTreeSet::new();
        for g in &record.grids {
            if !record.channels.contains(&g.channel) || !gridded.insert(g.channel) {
                return Err(invalid(format!(
                    "an action grid on {}, which the binding does not command once",
                    g.channel
                )));
            }
            if g.dt.is_zero() || g.dt > MAX_ACTION_DT {
                return Err(invalid(format!(
                    "the action period on {} is {:?}",
                    g.channel, g.dt
                )));
            }
        }
        Ok(())
    }

    /// Re-derives every stream's publication plan from the contracts of the
    /// attached controllers; a changed plan bumps its demand's revision.
    fn recompute_demands(&mut self, events: &mut Vec<AuthorityEvent>) {
        let mut wanted: BTreeMap<ChannelId, Vec<&crate::wire::StreamContract>> = BTreeMap::new();
        for (participant, binding) in &self.bindings {
            if !self.roster.contains_key(participant) {
                continue;
            }
            for c in &binding.contracts {
                wanted.entry(c.channel).or_default().push(c);
            }
        }
        let mut next: BTreeMap<ChannelId, (crate::wire::Plan, core::time::Duration)> =
            BTreeMap::new();
        for (channel, contracts) in wanted {
            if let Some((part, applied)) = self.stream_part(channel)
                && let Some(plan) = contract::plan(contracts, part, applied)
            {
                next.insert(channel, plan);
            }
        }
        let gone: Vec<ChannelId> = self
            .demands
            .keys()
            .filter(|c| !next.contains_key(c))
            .copied()
            .collect();
        for channel in gone {
            self.demands.remove(&channel);
            events.push(AuthorityEvent::DemandChanged {
                channel,
                demand: None,
            });
        }
        for (channel, (plan, retain)) in next {
            let unchanged = self
                .demands
                .get(&channel)
                .is_some_and(|d| d.plan == plan && d.retain == retain);
            if unchanged {
                continue;
            }
            let revision = self.demand_revisions.entry(channel).or_insert(0);
            *revision += 1;
            let acked = self.demands.get(&channel).map_or(0, |d| d.acked);
            let demand = Demand {
                channel,
                revision: *revision,
                plan,
                retain,
                acked,
            };
            self.demands.insert(channel, demand.clone());
            events.push(AuthorityEvent::DemandChanged {
                channel,
                demand: Some(demand),
            });
        }
        // Each attached binding's contracts are included from the current
        // revision on, unless an earlier one already included them.
        for (participant, binding) in &self.bindings {
            if !self.roster.contains_key(participant) {
                continue;
            }
            for c in &binding.contracts {
                if let Some(demand) = self.demands.get(&c.channel) {
                    self.needed
                        .entry((participant.clone(), c.channel))
                        .or_insert(demand.revision);
                }
            }
        }
    }

    fn demand_ack(
        &mut self,
        from: &ParticipantId,
        channel: ChannelId,
        revision: u64,
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<(), CtlError> {
        self.require_role(from, Role::Embodiment)?;
        let owner = self
            .stream_part(channel)
            .map(|(p, _)| p.owner.clone())
            .ok_or_else(|| {
                CtlError::new(
                    CtlStatus::UnknownPath,
                    format!("{channel} is not a declared stream"),
                )
            })?;
        if &owner != from {
            return Err(CtlError::new(
                CtlStatus::Forbidden,
                format!("{channel} is published by {owner}"),
            ));
        }
        let demand = self.demands.get_mut(&channel).ok_or_else(|| {
            CtlError::new(
                CtlStatus::InvalidRequest,
                format!("{channel} has no demand"),
            )
        })?;
        if revision > demand.revision {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                format!(
                    "{channel}: revision {revision} is ahead of the demand's {}",
                    demand.revision
                ),
            ));
        }
        if revision > demand.acked {
            demand.acked = revision;
            events.push(AuthorityEvent::DemandInstalled { channel, revision });
        }
        Ok(())
    }

    /// Validates the whole request, then grants: all or none.
    fn grant(
        &mut self,
        from: &ParticipantId,
        epoch: Option<Epoch>,
        grant: &Grant,
        now_ns: i64,
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<Vec<Lease>, CtlError> {
        let invalid = |message: String| CtlError::new(CtlStatus::InvalidRequest, message);
        let priority = self.require_role(from, Role::Controller)?.priority;
        if let Some(epoch) = epoch
            && epoch != self.epoch
        {
            return Err(CtlError::new(
                CtlStatus::StaleEpoch,
                format!("request epoch {epoch}, current {}", self.epoch),
            ));
        }
        match self.mode {
            Mode::Armed | Mode::Active => {}
            Mode::Estop => {
                return Err(CtlError::new(
                    CtlStatus::EstopLatched,
                    "session is in ESTOP",
                ));
            }
            Mode::Disarmed | Mode::Safe => {
                return Err(CtlError::new(
                    CtlStatus::InvalidTransition,
                    format!("session is {}; arm it first", self.mode),
                ));
            }
        }
        if grant.channels.is_empty() {
            return Err(invalid("a grant names no part".into()));
        }
        let mut named = BTreeSet::new();
        for channel in grant.channels.iter().chain(&grant.release) {
            if !named.insert(*channel) {
                return Err(invalid(format!("{channel} is named twice")));
            }
        }
        for channel in &grant.channels {
            if self.commandable(*channel).is_none() {
                return Err(invalid(format!("{channel} is not a commandable part")));
            }
        }
        let record = self.bindings.get(from).ok_or_else(|| {
            CtlError::new(
                CtlStatus::NotBound,
                format!("{from} has no binding: declare one first"),
            )
        })?;
        if record.fingerprint != grant.binding {
            return Err(CtlError::new(
                CtlStatus::NotBound,
                format!(
                    "binding {} is not {from}'s current binding {}",
                    grant.binding, record.fingerprint
                ),
            ));
        }
        if let Some(outside) = grant.channels.iter().find(|c| !record.channels.contains(c)) {
            return Err(CtlError::new(
                CtlStatus::NotBound,
                format!("{outside} is outside binding {}", record.fingerprint),
            ));
        }
        // Install before authority (§8.2).
        for c in &record.contracts {
            let needed = self.needed.get(&(from.clone(), c.channel)).copied();
            let installed = self
                .demands
                .get(&c.channel)
                .is_some_and(|d| needed.is_some_and(|n| d.installed(n)));
            if !installed {
                let path = self
                    .stream_part(c.channel)
                    .map(|(p, applied)| {
                        if applied {
                            p.path.applied()
                        } else {
                            p.path.to_string()
                        }
                    })
                    .unwrap_or_else(|| c.channel.to_string());
                return Err(CtlError::new(
                    CtlStatus::NotReady,
                    format!(
                        "{path} has not installed the delivery plan for {from}'s binding \
                         (demand revision {}); retry once its publisher acknowledges it",
                        needed.unwrap_or(0)
                    ),
                ));
            }
        }
        for channel in &grant.release {
            if self.leases.get(channel).is_none_or(|l| &l.holder != from) {
                return Err(invalid(format!("{channel} is not held by {from}")));
            }
        }
        for (i, a) in grant.channels.iter().enumerate() {
            if let Some(group) = self.group_of(*a)
                && grant.channels[i + 1..]
                    .iter()
                    .any(|b| self.group_of(*b) == Some(group))
            {
                return Err(invalid(format!(
                    "two members of command group {:?} in one request",
                    group.1
                )));
            }
        }
        // Who the grant takes over: (preempted part, the requested part).
        let mut preempted: Vec<(ChannelId, ChannelId)> = Vec::new();
        for channel in &grant.channels {
            match self.leases.get(channel) {
                Some(l) if &l.holder == from => {}
                Some(l) if l.priority < priority => {}
                Some(l) => {
                    return Err(CtlError::new(
                        CtlStatus::LeaseBusy,
                        format!(
                            "{channel} is held by {} at {:?}; {from} at {priority:?} cannot preempt it",
                            l.holder, l.priority
                        ),
                    ));
                }
                None => {}
            }
            for peer in self.group_peers(*channel) {
                if grant.release.contains(&peer) {
                    continue;
                }
                match self.leases.get(&peer) {
                    Some(l) if &l.holder == from => {
                        return Err(CtlError::new(
                            CtlStatus::GroupBusy,
                            format!(
                                "{from} holds {peer} in the same command group: release it in the same request"
                            ),
                        ));
                    }
                    Some(l) if l.priority < priority => preempted.push((peer, *channel)),
                    Some(l) => {
                        return Err(CtlError::new(
                            CtlStatus::GroupBusy,
                            format!(
                                "{peer}, in the command group of {channel}, is held by {} at {:?}",
                                l.holder, l.priority
                            ),
                        ));
                    }
                    None => {}
                }
            }
        }

        // Validated: mutate.
        let ttl_ns = grant
            .ttl_ns
            .filter(|ttl| *ttl > 0)
            .unwrap_or(self.config.lease_grace_ns);
        let expires_ns = now_ns.saturating_add_unsigned(ttl_ns);
        for channel in &grant.release {
            if let Some(lease) = self.leases.remove(channel) {
                let why = self.reason(ReasonCode::Released, from, "");
                self.end(lease, why, events);
            }
        }
        let needs_token = grant
            .channels
            .iter()
            .any(|c| self.leases.get(c).is_none_or(|l| &l.holder != from));
        let token = needs_token.then(|| self.next_token());
        let mut granted = Vec::with_capacity(grant.channels.len());
        for channel in &grant.channels {
            if let Some(lease) = self.leases.get_mut(channel)
                && &lease.holder == from
            {
                // A renewal under the same token (and binding).
                lease.ttl_ns = ttl_ns;
                lease.expires_ns = expires_ns;
                granted.push(lease.clone());
                continue;
            }
            let lease = Lease {
                channel: *channel,
                holder: from.clone(),
                priority,
                token: token.unwrap_or_default(),
                ttl_ns,
                granted_ns: now_ns,
                expires_ns,
                binding: grant.binding,
            };
            events.push(match self.leases.insert(*channel, lease.clone()) {
                Some(previous) => {
                    let why = self.reason(ReasonCode::Preempted, from, "");
                    self.record_end(&previous, why, Some(&lease));
                    AuthorityEvent::LeasePreempted {
                        previous,
                        by: lease.clone(),
                    }
                }
                None => AuthorityEvent::LeaseGranted {
                    lease: lease.clone(),
                },
            });
            granted.push(lease);
        }
        for (peer, by) in preempted {
            if let (Some(previous), Some(by)) =
                (self.leases.remove(&peer), self.leases.get(&by).cloned())
            {
                let why = self.reason(ReasonCode::Preempted, from, "");
                self.record_end(&previous, why, Some(&by));
                events.push(AuthorityEvent::LeasePreempted { previous, by });
            }
        }
        if self.mode == Mode::Armed && !self.leases.is_empty() {
            let why = self.reason(ReasonCode::FirstLease, from, "");
            self.set_mode(Mode::Active, why, events);
        }
        Ok(granted)
    }

    fn release(
        &mut self,
        from: &ParticipantId,
        channels: &[ChannelId],
        token: LeaseToken,
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<(), CtlError> {
        if channels.is_empty() {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                "a release names no part",
            ));
        }
        let unique: BTreeSet<ChannelId> = channels.iter().copied().collect();
        if unique.len() != channels.len() {
            return Err(CtlError::new(
                CtlStatus::InvalidRequest,
                "a release names a part twice",
            ));
        }
        for channel in channels {
            match self.leases.get(channel) {
                Some(lease) if &lease.holder == from && lease.token == token => {}
                Some(lease) => {
                    return Err(CtlError::new(
                        CtlStatus::StaleToken,
                        format!(
                            "{channel} is held by {} with {}, not {from} with {token}",
                            lease.holder, lease.token
                        ),
                    ));
                }
                None => {
                    return Err(CtlError::new(
                        CtlStatus::StaleToken,
                        format!("no lease on {channel}"),
                    ));
                }
            }
        }
        for channel in channels {
            if let Some(lease) = self.leases.remove(channel) {
                let why = self.reason(ReasonCode::Released, from, "");
                self.end(lease, why, events);
            }
        }
        self.settle_after_release(from, "last lease released", events);
        Ok(())
    }

    fn require_tick_verb(
        &self,
        from: &ParticipantId,
        epoch: Option<Epoch>,
    ) -> Result<(), CtlError> {
        self.require_attached(from)?;
        if self.clock.is_none() {
            return Err(CtlError::new(
                CtlStatus::Unsupported,
                "tick verbs need a virtual-clock session",
            ));
        }
        if epoch != Some(self.epoch) {
            return Err(CtlError::new(
                CtlStatus::StaleEpoch,
                format!(
                    "tick verb for epoch {}, current {}",
                    epoch.unwrap_or_default(),
                    self.epoch
                ),
            ));
        }
        Ok(())
    }

    /// Who the virtual barrier waits for.
    fn barrier_participants(&self) -> Participants {
        let tick_ns = self.config.virtual_clock.map_or(1, |c| c.tick_ns.max(1));
        Participants {
            embodiments: self
                .parts
                .values()
                .map(|p| &p.owner)
                .filter(|owner| self.roster.contains_key(*owner))
                .cloned()
                .collect(),
            controllers: self
                .bindings
                .values()
                .filter(|b| self.roster.contains_key(&b.participant) && b.replan_hz > 0.0)
                .map(|b| {
                    let period = (1e9 / (b.replan_hz * tick_ns as f64)).round().max(1.0);
                    (b.participant.clone(), period as u64)
                })
                .collect(),
        }
    }

    fn barrier_gone(&mut self, id: &ParticipantId, now_ns: i64) {
        let who = self.barrier_participants();
        if let Some(clock) = &mut self.clock {
            let ticks = clock.on_gone(id, &who, now_ns);
            self.ticks.extend(ticks);
        }
    }

    /// A reason for a change landing in this call's revision.
    fn reason(&self, code: ReasonCode, by: impl fmt::Display, detail: &str) -> Reason {
        Reason {
            code,
            by: by.to_string(),
            detail: detail.to_string(),
            at_ns: self.now_ns,
            revision: self.revision + 1,
            verified: false,
        }
    }

    /// Changes the mode and records why; an entry into SAFE, DISARMED or
    /// ESTOP moves `last_stop` (never lowered: epochs and revisions only
    /// grow).
    fn set_mode(&mut self, to: Mode, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        if self.mode == to {
            return;
        }
        if matches!(to, Mode::Safe | Mode::Disarmed | Mode::Estop) {
            self.last_stop = LastStop {
                epoch: self.epoch,
                revision: self.revision + 1,
            };
        }
        events.push(AuthorityEvent::ModeChanged {
            from: self.mode,
            to,
            epoch: self.epoch,
            reason: reason.clone(),
        });
        self.mode = to;
        self.mode_reason = reason;
    }

    /// Keeps a lease ending in the snapshot's ring.
    fn record_end(&mut self, lease: &Lease, reason: Reason, by: Option<&Lease>) {
        if self.ended.len() == ENDED {
            self.ended.pop_front();
        }
        self.ended.push_back(LeaseEnd {
            lease: lease.clone(),
            reason,
            by: by.cloned(),
        });
    }

    /// A lease ended (not by a preemption or a rebinding, which have
    /// events of their own).
    fn end(&mut self, lease: Lease, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        self.record_end(&lease, reason.clone(), None);
        events.push(AuthorityEvent::LeaseEnded { lease, reason });
    }

    fn end_leases_of(
        &mut self,
        id: &ParticipantId,
        reason: &Reason,
        events: &mut Vec<AuthorityEvent>,
    ) {
        let channels: Vec<ChannelId> = self
            .leases
            .values()
            .filter(|lease| &lease.holder == id)
            .map(|lease| lease.channel)
            .collect();
        for channel in channels {
            if let Some(lease) = self.leases.remove(&channel) {
                self.end(lease, reason.clone(), events);
            }
        }
    }

    /// Ends every lease with `code`, carrying who and why from `cause`.
    fn revoke_all(&mut self, code: ReasonCode, cause: &Reason, events: &mut Vec<AuthorityEvent>) {
        for (_, lease) in core::mem::take(&mut self.leases) {
            let why = Reason {
                code,
                ..cause.clone()
            };
            self.end(lease, why, events);
        }
    }

    /// ACTIVE with no lease left drops back to ARMED.
    fn settle_after_release(
        &mut self,
        by: &ParticipantId,
        detail: &str,
        events: &mut Vec<AuthorityEvent>,
    ) {
        if self.mode == Mode::Active && self.leases.is_empty() {
            let why = self.reason(ReasonCode::LastRelease, by, detail);
            self.set_mode(Mode::Armed, why, events);
        }
    }

    /// A lapse while ACTIVE takes the whole session to SAFE and revokes every
    /// other lease: re-arming (a fresh epoch) is the way back.
    fn enter_safe(&mut self, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        if self.mode == Mode::Active {
            self.revoke_all(ReasonCode::VoidedBySafe, &reason, events);
            self.set_mode(Mode::Safe, reason, events);
        }
    }
}

/// A `GrantLease` verb's arguments.
struct Grant {
    channels: Vec<ChannelId>,
    release: Vec<ChannelId>,
    binding: Fingerprint,
    ttl_ns: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::version::SPEC_VERSION;

    const GRACE: u64 = 100;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    /// Four ungrouped commandable parts, an `arm` group of two, and a
    /// sensor.
    fn robot_map(extra: &str) -> String {
        let cmd = |path: &str, group: &str| {
            format!(
                r#""{path}": {{"kind": "joints", "hz": 500, "count": 6, "fields": ["pos"],
                   "command": "position"{group}}}"#
            )
        };
        let parts = [
            cmd("cmd1", ""),
            cmd("cmd2", ""),
            cmd("cmd3", ""),
            cmd("cmd4", ""),
            cmd("arm", r#", "group": "arm""#),
            cmd("arm/tcp", r#", "group": "arm""#),
            r#""ft": {"kind": "x/ft", "hz": 1000, "fields": {"wrench": ["float32", [6]]}}"#
                .to_string(),
        ];
        format!("{{{}{extra}}}", parts.join(","))
    }

    struct Fixture {
        auth: Authority,
    }

    impl Fixture {
        /// robot (embodiment), policy (model), operator, script, viewer; each
        /// controller bound to every commandable part.
        fn new() -> Self {
            let mut auth = Authority::new(
                SessionId::new(),
                AuthorityConfig {
                    lease_grace_ns: GRACE,
                    ..AuthorityConfig::default()
                },
            );
            for (id, roles, priority) in [
                ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
                ("policy", vec![Role::Controller], PriorityClass::Model),
                ("operator", vec![Role::Controller], PriorityClass::Operator),
                ("script", vec![Role::Controller], PriorityClass::Scripted),
                ("viewer", vec![Role::Observer], PriorityClass::Scripted),
            ] {
                auth.apply(
                    Request {
                        from: pid(id),
                        epoch: None,
                        command: attach(id, roles, priority),
                    },
                    0,
                )
                .expect("attach");
            }
            let mut f = Self { auth };
            f.run("robot", declare(&robot_map("")), 0).expect("declare");
            for (i, id) in ["policy", "operator", "script"].into_iter().enumerate() {
                f.bind(
                    id,
                    i as u64 + 1,
                    &["cmd1", "cmd2", "cmd3", "cmd4", "arm", "arm/tcp"],
                )
                .expect("bind");
            }
            f
        }

        fn ch(&self, path: &str) -> ChannelId {
            self.auth
                .parts()
                .find(|p| p.path.as_str() == path)
                .map(|p| p.channel)
                .expect(path)
        }

        fn run(&mut self, from: &str, command: Command, now: i64) -> Result<Applied, CtlError> {
            let epoch = Some(self.auth.epoch());
            self.auth.apply(
                Request {
                    from: pid(from),
                    epoch,
                    command,
                },
                now,
            )
        }

        fn bind(&mut self, from: &str, fp: u64, paths: &[&str]) -> Result<Applied, CtlError> {
            let channels = paths.iter().map(|p| self.ch(p)).collect();
            self.run(
                from,
                Command::DeclareBinding {
                    fingerprint: Fingerprint::from(fp),
                    revision: 1,
                    table_json: "{}".into(),
                    channels,
                    replan_hz: 0.0,
                    contracts: Vec::new(),
                    grids: Vec::new(),
                },
                0,
            )
        }

        fn fingerprint(&self, from: &str) -> Fingerprint {
            self.auth
                .binding(&pid(from))
                .map_or_else(Fingerprint::default, |b| b.fingerprint)
        }

        fn grant_with(
            &mut self,
            from: &str,
            paths: &[&str],
            release: &[&str],
            now: i64,
        ) -> Result<Vec<Lease>, CtlError> {
            let command = Command::GrantLease {
                channels: paths.iter().map(|p| self.ch(p)).collect(),
                release: release.iter().map(|p| self.ch(p)).collect(),
                binding: self.fingerprint(from),
                ttl_ns: None,
            };
            self.run(from, command, now).map(|a| a.leases)
        }

        fn grant(&mut self, from: &str, path: &str, now: i64) -> Result<Lease, CtlError> {
            self.grant_with(from, &[path], &[], now)
                .map(|mut leases| leases.remove(0))
        }

        fn release(&mut self, from: &str, lease: &Lease, now: i64) -> Result<Applied, CtlError> {
            self.run(
                from,
                Command::ReleaseLease {
                    channels: vec![lease.channel],
                    token: lease.token,
                },
                now,
            )
        }

        fn status(result: Result<impl fmt::Debug, CtlError>) -> CtlStatus {
            result.expect_err("refused").status
        }
    }

    fn attach(name: &str, roles: Vec<Role>, priority: PriorityClass) -> Command {
        Command::Attach {
            name: name.into(),
            roles,
            priority,
            protocol_minor: PROTOCOL_MINOR,
            supports: Vec::new(),
            requires: Vec::new(),
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
        }
    }

    fn declare(json: &str) -> Command {
        Command::DeclareParts {
            spec_version: SPEC_VERSION,
            robot_json: json.into(),
        }
    }

    #[test]
    fn arm_bumps_the_epoch_only_from_disarmed_or_safe() {
        let mut f = Fixture::new();
        assert_eq!(f.auth.epoch(), Epoch(1));
        f.run("operator", Command::Arm, 1).expect("arm");
        assert_eq!((f.auth.mode(), f.auth.epoch()), (Mode::Armed, Epoch(2)));
        assert_eq!(
            Fixture::status(f.run("operator", Command::Arm, 2)),
            CtlStatus::InvalidTransition
        );
        f.run("operator", Command::Disarm, 3).expect("disarm");
        f.run("operator", Command::Arm, 4).expect("re-arm");
        assert_eq!(f.auth.epoch(), Epoch(3));
    }

    #[test]
    fn verbs_need_an_attached_participant() {
        let mut f = Fixture::new();
        assert_eq!(
            Fixture::status(f.run("stranger", Command::Arm, 0)),
            CtlStatus::NotAttached
        );
    }

    #[test]
    fn the_host_assigns_part_ids_and_never_reuses_them() {
        let mut f = Fixture::new();
        let parts: Vec<&PartSpec> = f.auth.parts().collect();
        assert_eq!(parts.len(), 7);
        let arm = f.auth.part(f.ch("arm")).expect("arm").clone();
        let applied = arm
            .applied_channel
            .expect("commandable parts get an applied id");
        assert_ne!(applied, arm.channel);
        assert!(
            f.auth
                .part(f.ch("ft"))
                .expect("ft")
                .applied_channel
                .is_none()
        );
        let mut ids: Vec<u32> = f
            .auth
            .parts()
            .flat_map(|p| [Some(p.channel), p.applied_channel])
            .flatten()
            .map(|c| c.0)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 7 + 6, "every stream id is distinct");

        // The snapshot carries the declaration fingerprint (§3.3).
        let contract = |f: &Fixture| {
            f.auth
                .snapshot(Vec::new(), 0)
                .contract
                .get(&pid("robot"))
                .copied()
        };
        let declared = contract(&f).expect("a declaration fingerprint");

        // An identical re-declaration is a no-op.
        let revision = f.auth.revision();
        f.run("robot", declare(&robot_map("")), 1)
            .expect("redeclare");
        assert_eq!(f.auth.revision(), revision);
        assert_eq!(f.ch("arm"), arm.channel);
        assert_eq!(contract(&f), Some(declared));

        // A changed schema retires the id; armed, only sensors may change.
        f.run("operator", Command::Arm, 1).expect("arm");
        let ft = f.ch("ft");
        let hot =
            robot_map(r#", "wrist/cam": {"kind": "camera", "hz": 30, "width": 4, "height": 4}"#)
                .replace(r#""hz": 1000"#, r#""hz": 500"#);
        let applied = f.run("robot", declare(&hot), 2).expect("hot-plug");
        assert_ne!(
            contract(&f),
            Some(declared),
            "a new schema, a new fingerprint"
        );
        assert!(matches!(
            applied.events.as_slice(),
            [AuthorityEvent::PartsRemoved { channels, .. }, AuthorityEvent::PartsDeclared { .. }]
                if channels == &[ft]
        ));
        assert!(
            f.ch("ft") > ft && f.ch("wrist/cam") > ft,
            "fresh ids, never reused"
        );
        let changed = robot_map("").replace(
            r#""cmd1": {"kind": "joints", "hz": 500"#,
            r#""cmd1": {"kind": "joints", "hz": 250"#,
        );
        assert_eq!(
            Fixture::status(f.run("robot", declare(&changed), 3)),
            CtlStatus::InvalidTransition,
            "a commandable part is not re-declared while armed"
        );
        f.run("operator", Command::Disarm, 4).expect("disarm");
        let cmd1 = f.ch("cmd1");
        f.run("robot", declare(&changed), 5).expect("disarmed");
        assert!(f.ch("cmd1") > cmd1);
        assert!(f.auth.part(cmd1).is_none());

        // Only embodiments declare parts; a bad map changes nothing.
        assert_eq!(
            Fixture::status(f.run("viewer", declare(&robot_map("")), 6)),
            CtlStatus::Forbidden
        );
        let before = f.auth.parts().count();
        assert_eq!(
            Fixture::status(f.run("robot", declare(r#"{"Bad": {}}"#), 6)),
            CtlStatus::InvalidRequest
        );
        assert_eq!(f.auth.parts().count(), before);
    }

    #[test]
    fn first_lease_activates_and_releasing_the_last_rearms() {
        let mut f = Fixture::new();
        assert_eq!(
            Fixture::status(f.grant("policy", "cmd4", 0)),
            CtlStatus::InvalidTransition,
            "grant before arm"
        );
        f.run("operator", Command::Arm, 0).expect("arm");
        assert_eq!(
            Fixture::status(f.grant("viewer", "cmd4", 0)),
            CtlStatus::Forbidden
        );
        let lease = f.grant("policy", "cmd4", 10).expect("grant");
        assert_eq!(f.auth.mode(), Mode::Active);
        assert_eq!(lease.expires_ns, 10 + GRACE as i64);
        assert_eq!(lease.binding, f.fingerprint("policy"));
        f.auth
            .check_fence(f.ch("cmd4"), lease.token, f.auth.epoch(), 20)
            .expect("fence");

        let revision = f.auth.revision();
        let renewed = f.grant("policy", "cmd4", 50).expect("re-grant");
        assert_eq!(renewed.token, lease.token, "renewal keeps the token");
        assert_eq!(f.auth.revision(), revision, "renewal is not structural");

        f.release("policy", &lease, 60).expect("release");
        assert_eq!(f.auth.mode(), Mode::Armed);
        assert!(f.auth.lease(f.ch("cmd4")).is_none());
    }

    #[test]
    fn a_multi_part_grant_is_all_or_none() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let script = f.grant("script", "cmd3", 0).expect("script lease");
        f.grant("operator", "cmd2", 0).expect("operator lease");
        // cmd2 is held by a higher class: nothing of the request is granted,
        // not even the free cmd1, and the preemptible cmd3 is untouched.
        assert_eq!(
            Fixture::status(f.grant_with("policy", &["cmd1", "cmd3", "cmd2"], &[], 1)),
            CtlStatus::LeaseBusy
        );
        assert!(f.auth.lease(f.ch("cmd1")).is_none());
        assert_eq!(
            f.auth.lease(f.ch("cmd3")).map(|l| l.token),
            Some(script.token)
        );

        let leases = f
            .grant_with("policy", &["cmd1", "cmd3"], &[], 2)
            .expect("all of it");
        assert_eq!(leases.len(), 2);
        assert_eq!(leases[0].token, leases[1].token, "one token per request");
        assert!(leases[0].token > script.token);

        let cmd1 = f.ch("cmd1");
        for (paths, release) in [
            (vec!["cmd4", "cmd4"], vec![]),
            (vec!["cmd4"], vec!["cmd4"]),
            (vec!["cmd4"], vec!["cmd2"]),
            (vec!["ft"], vec![]),
            (vec![], vec![]),
        ] {
            assert_eq!(
                Fixture::status(f.grant_with("policy", &paths, &release, 3)),
                CtlStatus::InvalidRequest,
                "{paths:?} releasing {release:?}"
            );
        }
        assert_eq!(
            f.auth.lease(cmd1).map(|l| l.holder.clone()),
            Some(pid("policy"))
        );
    }

    #[test]
    fn a_lease_needs_the_requesters_current_binding() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.bind("script", 30, &["cmd1"]).expect("narrow binding");
        assert_eq!(
            Fixture::status(f.grant("script", "cmd2", 0)),
            CtlStatus::NotBound,
            "outside the binding"
        );
        let stale = Command::GrantLease {
            channels: vec![f.ch("cmd1")],
            release: Vec::new(),
            binding: Fingerprint::from(3),
            ttl_ns: None,
        };
        assert_eq!(
            Fixture::status(f.run("script", stale, 0)),
            CtlStatus::NotBound,
            "a binding that is not current"
        );
        f.run(
            "robot",
            attach("x", vec![Role::Controller], PriorityClass::Model),
            0,
        )
        .expect("re-attach as a controller");
        let unbound = Command::GrantLease {
            channels: vec![f.ch("cmd1")],
            release: Vec::new(),
            binding: Fingerprint::from(1),
            ttl_ns: None,
        };
        assert_eq!(
            Fixture::status(f.run("robot", unbound, 0)),
            CtlStatus::NotBound
        );
        assert_eq!(
            Fixture::status(f.bind("script", 31, &["ft"])),
            CtlStatus::UnknownPath
        );
        assert_eq!(
            Fixture::status(f.bind("script", 31, &["cmd1", "cmd1"])),
            CtlStatus::InvalidRequest
        );
        assert_eq!(
            Fixture::status(f.bind("script", 30, &["cmd2"])),
            CtlStatus::InvalidRequest,
            "one fingerprint, one binding"
        );
        assert_eq!(
            Fixture::status(f.bind("viewer", 40, &[])),
            CtlStatus::Forbidden
        );
    }

    fn contract(
        channel: ChannelId,
        mode: crate::wire::ContractMode,
        delivery_hz: f64,
        history: u32,
        max_age_ms: Option<u64>,
    ) -> crate::wire::StreamContract {
        crate::wire::StreamContract {
            channel,
            mode,
            delivery_hz,
            history,
            max_age: Some(core::time::Duration::from_millis(max_age_ms.unwrap_or(150))),
            max_age_explicit: max_age_ms.is_some(),
        }
    }

    impl Fixture {
        fn bind_with(
            &mut self,
            from: &str,
            fp: u64,
            paths: &[&str],
            contracts: Vec<crate::wire::StreamContract>,
        ) -> Result<Applied, CtlError> {
            let channels: Vec<ChannelId> = paths.iter().map(|p| self.ch(p)).collect();
            let grids = channels
                .iter()
                .map(|c| crate::wire::ActionGrid {
                    channel: *c,
                    dt: core::time::Duration::from_millis(10),
                })
                .collect();
            self.run(
                from,
                Command::DeclareBinding {
                    fingerprint: Fingerprint::from(fp),
                    revision: 2,
                    table_json: "{}".into(),
                    channels,
                    replan_hz: 20.0,
                    contracts,
                    grids,
                },
                0,
            )
        }

        fn demand(&self, path: &str) -> Option<Demand> {
            self.auth.demands.get(&self.ch(path)).cloned()
        }

        fn ack(&mut self, path: &str, revision: u64) -> Result<Applied, CtlError> {
            let channel = self.ch(path);
            self.run("robot", Command::DemandAck { channel, revision }, 0)
        }
    }

    #[test]
    fn a_lease_waits_until_every_bound_stream_installed_its_plan() {
        use crate::wire::{ContractMode, Plan};
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let ft = f.ch("ft");
        let window = contract(ft, ContractMode::History, 20.0, 50, None);
        f.bind_with("policy", 20, &["cmd1"], vec![window])
            .expect("bind");
        let demand = f.demand("ft").expect("a demand on ft");
        assert_eq!((demand.revision, demand.acked), (1, 0));
        assert_eq!(demand.plan, Plan::Every, "a window needs every sample");
        let err = f.grant("policy", "cmd1", 0).expect_err("not installed");
        assert_eq!(err.status, CtlStatus::NotReady);
        assert!(
            err.message.contains("ft has not installed"),
            "{}",
            err.message
        );
        assert_eq!(
            Fixture::status(f.run(
                "policy",
                Command::DemandAck {
                    channel: ft,
                    revision: 1
                },
                0
            )),
            CtlStatus::Forbidden,
            "only the stream's publisher acknowledges"
        );
        assert_eq!(Fixture::status(f.ack("ft", 2)), CtlStatus::InvalidRequest);
        let applied = f.ack("ft", 1).expect("ack");
        assert!(matches!(
            applied.events.as_slice(),
            [AuthorityEvent::DemandInstalled { revision: 1, .. }]
        ));
        let lease = f.grant("policy", "cmd1", 0).expect("installed: granted");
        assert_eq!(
            f.auth.snapshot(Vec::new(), 0).demand(ft).map(|d| d.acked),
            Some(1)
        );

        // A second controller wants a deeper window: the demand (and its
        // revision) changes, but the first binding stays served by
        // revision 1.
        let deeper = contract(ft, ContractMode::History, 20.0, 100, None);
        f.bind_with("operator", 21, &["cmd2"], vec![deeper])
            .expect("bind");
        let demand = f.demand("ft").expect("demand");
        assert_eq!((demand.revision, demand.acked), (2, 1));
        assert_eq!(demand.plan, Plan::Every);
        assert_eq!(
            f.grant("policy", "cmd1", 0).expect("renewal").token,
            lease.token
        );
        assert_eq!(
            Fixture::status(f.grant("operator", "cmd2", 0)),
            CtlStatus::NotReady
        );
        f.ack("ft", 2).expect("ack");
        f.grant("operator", "cmd2", 0).expect("granted");

        // A controller that goes away takes its contract along; the last
        // one withdraws the demand.
        f.auth.participant_lost(&pid("operator"), 1);
        assert_eq!(f.demand("ft").map(|d| d.revision), Some(3));
        f.run("policy", Command::Detach, 2).expect("detach");
        assert_eq!(f.demand("ft"), None);
    }

    #[test]
    fn contracts_a_stream_cannot_meet_are_refused_at_admission() {
        use crate::wire::ContractMode;
        let mut f = Fixture::new();
        let (ft, cmd1) = (f.ch("ft"), f.ch("cmd1"));
        for (contracts, status, needle) in [
            (
                vec![contract(ft, ContractMode::Latest, 20.0, 1, Some(5))],
                CtlStatus::InvalidRequest,
                "below the 50 ms",
            ),
            (
                vec![contract(
                    ChannelId(999),
                    ContractMode::Latest,
                    20.0,
                    1,
                    None,
                )],
                CtlStatus::UnknownPath,
                "not a declared stream",
            ),
            (
                vec![
                    contract(ft, ContractMode::Latest, 20.0, 1, None),
                    contract(ft, ContractMode::Latest, 30.0, 1, None),
                ],
                CtlStatus::InvalidRequest,
                "two contracts",
            ),
            (
                vec![contract(ft, ContractMode::Latest, 20.0, 5, None)],
                CtlStatus::InvalidRequest,
                "no window",
            ),
        ] {
            let err = f
                .bind_with("policy", 30, &["cmd1"], contracts)
                .expect_err("refused");
            assert_eq!(err.status, status, "{}", err.message);
            assert!(err.message.contains(needle), "{}", err.message);
        }
        // A window on a commandable part's applied stream is a stream too.
        let applied = f
            .auth
            .part(cmd1)
            .and_then(|p| p.applied_channel)
            .expect("applied");
        f.bind_with(
            "policy",
            31,
            &["cmd1"],
            vec![contract(applied, ContractMode::Latest, 20.0, 1, None)],
        )
        .expect("an applied stream");
        assert!(f.auth.demands.contains_key(&applied));
        // An action grid on a part the binding does not command.
        let off_grid = Command::DeclareBinding {
            fingerprint: Fingerprint::from(33),
            revision: 3,
            table_json: "{}".into(),
            channels: vec![cmd1],
            replan_hz: 20.0,
            contracts: Vec::new(),
            grids: vec![crate::wire::ActionGrid {
                channel: f.ch("cmd2"),
                dt: core::time::Duration::from_millis(10),
            }],
        };
        assert_eq!(
            Fixture::status(f.run("policy", off_grid, 0)),
            CtlStatus::InvalidRequest
        );
        assert_eq!(
            f.auth.binding(&pid("policy")).and_then(|b| b.grid(cmd1)),
            Some(core::time::Duration::from_millis(10))
        );
    }

    #[test]
    fn removing_a_stream_withdraws_its_demand() {
        use crate::wire::ContractMode;
        let mut f = Fixture::new();
        let ft = f.ch("ft");
        f.bind_with(
            "policy",
            40,
            &["cmd1"],
            vec![contract(ft, ContractMode::Every, 1000.0, 1, Some(3))],
        )
        .expect("bind");
        assert_eq!(
            f.demand("ft").map(|d| d.plan),
            Some(crate::wire::Plan::Every)
        );
        let applied = f.run("robot", Command::Detach, 1).expect("detach");
        assert!(applied.events.iter().any(|e| matches!(
            e,
            AuthorityEvent::DemandChanged { channel, demand: None } if *channel == ft
        )));
        assert!(f.auth.demands.is_empty());
    }

    #[test]
    fn rebinding_reissues_the_holders_leases_under_a_fresh_token() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let old = f
            .grant_with("policy", &["cmd1", "cmd2"], &[], 0)
            .expect("leases");
        let epoch = f.auth.epoch();
        // The same binding again changes nothing.
        let applied = f
            .bind(
                "policy",
                1,
                &["cmd1", "cmd2", "cmd3", "cmd4", "arm", "arm/tcp"],
            )
            .expect("same binding");
        assert!(applied.events.is_empty());
        // A new binding keeps cmd1 under a new token and drops cmd2.
        let applied = f.bind("policy", 11, &["cmd1"]).expect("rebind");
        let rebound = f.auth.lease(f.ch("cmd1")).expect("kept").clone();
        assert!(rebound.token > old[0].token, "a fresh token");
        assert_eq!(rebound.binding, Fingerprint::from(11));
        assert!(
            f.auth.lease(f.ch("cmd2")).is_none(),
            "outside the new binding"
        );
        assert!(applied.events.iter().any(|e| matches!(
            e,
            AuthorityEvent::LeaseRebound { previous, .. } if previous.token == old[0].token
        )));
        assert!(matches!(
            f.auth.check_fence(f.ch("cmd1"), old[0].token, epoch, 1),
            Err(FenceError::StaleToken { .. })
        ));
        assert_eq!(f.auth.mode(), Mode::Active);
        // Dropping every part settles back to ARMED.
        f.bind("policy", 12, &["cmd4"]).expect("rebind");
        assert_eq!(f.auth.mode(), Mode::Armed);
    }

    #[test]
    fn command_groups_hold_one_member_and_switch_atomically() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        assert_eq!(
            Fixture::status(f.grant_with("policy", &["arm", "arm/tcp"], &[], 0)),
            CtlStatus::InvalidRequest,
            "two members of one group in one request"
        );
        let joint = f.grant("policy", "arm", 0).expect("arm");
        assert_eq!(
            Fixture::status(f.grant("policy", "arm/tcp", 1)),
            CtlStatus::GroupBusy,
            "holding a member without releasing it"
        );
        assert_eq!(
            Fixture::status(f.grant("script", "arm/tcp", 1)),
            CtlStatus::GroupBusy,
            "a member held by a higher class"
        );
        let applied = f
            .run(
                "policy",
                Command::GrantLease {
                    channels: vec![f.ch("arm/tcp")],
                    release: vec![f.ch("arm")],
                    binding: f.fingerprint("policy"),
                    ttl_ns: None,
                },
                2,
            )
            .expect("switch");
        let tcp = applied.leases[0].clone();
        assert!(tcp.token > joint.token);
        assert!(f.auth.lease(f.ch("arm")).is_none());
        let revision = f.auth.revision();

        // The operator takes the other member: the model's is preempted in
        // the same revision.
        let human = f
            .grant("operator", "arm", 3)
            .expect("preempt across the group");
        assert_eq!(f.auth.revision(), revision + 1);
        assert!(f.auth.lease(f.ch("arm/tcp")).is_none());
        assert_eq!(
            f.auth.lease(f.ch("arm")).map(|l| l.token),
            Some(human.token)
        );
        assert_eq!(
            Fixture::status(f.grant("policy", "arm/tcp", 4)),
            CtlStatus::GroupBusy
        );
        // Other parts are independent of the group.
        f.grant("policy", "cmd1", 4).expect("ungrouped");
    }

    #[test]
    fn a_grant_from_a_stale_epoch_is_refused() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let result = f.auth.apply(
            Request {
                from: pid("policy"),
                epoch: Some(Epoch(1)),
                command: Command::GrantLease {
                    channels: vec![f.ch("cmd1")],
                    release: Vec::new(),
                    binding: f.fingerprint("policy"),
                    ttl_ns: None,
                },
            },
            0,
        );
        assert_eq!(Fixture::status(result), CtlStatus::StaleEpoch);
    }

    #[test]
    fn operator_preempts_model_and_the_old_token_is_fenced() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let model = f.grant("policy", "cmd1", 0).expect("model lease");
        let human = f.grant("operator", "cmd1", 5).expect("preempt");
        assert!(human.token > model.token, "preemption bumps the token");
        assert_eq!(f.auth.mode(), Mode::Active);

        let epoch = f.auth.epoch();
        let cmd1 = f.ch("cmd1");
        assert_eq!(
            f.auth.check_fence(cmd1, model.token, epoch, 6),
            Err(FenceError::StaleToken {
                channel: cmd1,
                current: human.token,
                got: model.token
            })
        );
        f.auth
            .check_fence(cmd1, human.token, epoch, 6)
            .expect("new token");
        assert_eq!(
            Fixture::status(f.release("policy", &model, 7)),
            CtlStatus::StaleToken
        );
        assert_eq!(
            Fixture::status(f.grant("policy", "cmd1", 8)),
            CtlStatus::LeaseBusy
        );
        f.release("operator", &human, 9).expect("release");
        let again = f.grant("policy", "cmd1", 10).expect("model again");
        assert!(again.token > human.token);
        // The snapshot says why each lease ended, and who took it.
        let ended = f.auth.snapshot(Vec::new(), 10).ended;
        let why: Vec<_> = ended
            .iter()
            .map(|e| (e.lease.token, e.reason.code, e.by.as_ref().map(|b| b.token)))
            .collect();
        assert_eq!(
            why,
            [
                (model.token, ReasonCode::Preempted, Some(human.token)),
                (human.token, ReasonCode::Released, None),
            ]
        );
        assert_eq!(f.auth.mode_reason().code, ReasonCode::FirstLease);
    }

    #[test]
    fn equal_or_lower_classes_cannot_preempt() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", "cmd1", 0).expect("lease");
        assert_eq!(
            Fixture::status(f.grant("script", "cmd1", 1)),
            CtlStatus::LeaseBusy
        );
        f.grant("script", "cmd2", 1).expect("other part");
    }

    #[test]
    fn an_embodiments_fallback_goes_safe_from_any_mode_and_estop_records_it() {
        let mut f = Fixture::new();
        let safe = |code| Command::EnterSafe {
            reason: "board link".into(),
            code,
        };
        let stopped_now = |f: &Fixture| {
            assert_eq!(
                f.auth.last_stop(),
                LastStop {
                    epoch: f.auth.epoch(),
                    revision: f.auth.revision()
                }
            );
        };
        // Between episodes too: DISARMED and ARMED go SAFE, so a device
        // fault is visible at once and the next drive needs an arm.
        f.run("robot", safe(ReasonCode::DeviceWatchdog), 1)
            .expect("from DISARMED");
        assert_eq!(f.auth.mode(), Mode::Safe);
        let reason = f.auth.mode_reason();
        assert_eq!(
            (reason.code, reason.by.as_str(), reason.revision),
            (ReasonCode::DeviceWatchdog, "robot", f.auth.revision())
        );
        stopped_now(&f);
        f.run("operator", Command::Arm, 2).expect("arm");
        assert_eq!(f.auth.last_stop().epoch.next(), f.auth.epoch(), "kept");
        f.run("robot", safe(ReasonCode::DeviceWatchdog), 3)
            .expect("from ARMED");
        assert_eq!(f.auth.mode(), Mode::Safe);
        stopped_now(&f);
        f.run("operator", Command::Arm, 4).expect("arm");
        f.grant("policy", "cmd1", 4).expect("lease");
        assert_eq!(
            Fixture::status(f.run("policy", safe(ReasonCode::QueueDry), 5)),
            CtlStatus::Forbidden
        );
        f.run("robot", safe(ReasonCode::QueueDry), 6)
            .expect("from ACTIVE");
        assert_eq!(f.auth.mode(), Mode::Safe);
        stopped_now(&f);
        assert!(f.auth.lease(f.ch("cmd1")).is_none(), "SAFE revokes leases");
        let ended = f.auth.snapshot(Vec::new(), 6).ended;
        assert_eq!(
            ended.last().map(|e| (e.reason.code, e.reason.by.as_str())),
            Some((ReasonCode::VoidedBySafe, "robot"))
        );
        // ESTOP stays ESTOP and records the cause; recording is no new stop.
        let estop = Command::Estop {
            reason: "button".into(),
            code: ReasonCode::Unspecified,
        };
        f.run("viewer", estop, 7).expect("estop");
        assert_eq!(f.auth.mode_reason().code, ReasonCode::EstopRemote);
        stopped_now(&f);
        let stop = f.auth.last_stop();
        f.run("robot", safe(ReasonCode::DeviceWatchdog), 8)
            .expect("recorded");
        assert_eq!(
            (f.auth.mode(), f.auth.mode_reason().code, f.auth.last_stop()),
            (Mode::Estop, ReasonCode::DeviceWatchdog, stop)
        );
        assert!(f.auth.estop_latched());
    }

    #[test]
    fn heartbeats_renew_and_their_absence_lapses_to_safe() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let a = f.grant("policy", "cmd1", 0).expect("lease");
        f.grant("script", "cmd2", 0).expect("lease");
        let epoch = f.auth.epoch();
        for now in [50, 100, 150] {
            f.auth.heartbeat(&pid("policy"), epoch, now);
            f.auth.heartbeat(&pid("script"), epoch, now);
            assert!(f.auth.expire(now).is_empty());
        }
        f.auth.heartbeat(&pid("policy"), Epoch(1), 200);
        f.auth.heartbeat(&pid("script"), epoch, 200);
        let events = f.auth.expire(251);
        assert_eq!(f.auth.mode(), Mode::Safe);
        assert!(events.iter().any(|e| matches!(
            e,
            AuthorityEvent::LeaseEnded { lease, reason }
                if lease.token == a.token && reason.code == ReasonCode::HeartbeatLapse
        )));
        let reason = f.auth.mode_reason();
        assert_eq!(
            (reason.code, reason.by.as_str()),
            (ReasonCode::HeartbeatLapse, "policy")
        );
        assert!(f.auth.lease(f.ch("cmd2")).is_none());
        assert_eq!(
            Fixture::status(f.grant("policy", "cmd1", 260)),
            CtlStatus::InvalidTransition
        );
        f.run("operator", Command::Arm, 270)
            .expect("re-arm from SAFE");
        assert_eq!(f.auth.epoch(), epoch.next());
    }

    #[test]
    fn losing_the_holder_goes_safe_and_losing_an_observer_does_not() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", "cmd1", 0).expect("lease");
        f.auth.participant_lost(&pid("viewer"), 0);
        assert_eq!(f.auth.mode(), Mode::Active);
        f.auth.participant_lost(&pid("policy"), 0);
        assert_eq!(f.auth.mode(), Mode::Safe);
        assert!(!f.auth.is_attached(&pid("policy")));
        assert!(
            f.auth.binding(&pid("policy")).is_some(),
            "kept for a reconnect"
        );
    }

    #[test]
    fn a_detaching_owner_takes_its_parts_and_their_leases_along() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", "cmd1", 0).expect("lease");
        let applied = f.run("robot", Command::Detach, 1).expect("detach");
        assert!(applied.events.iter().any(|e| matches!(
            e,
            AuthorityEvent::LeaseEnded { reason, .. } if reason.code == ReasonCode::PartRemoved
        )));
        assert_eq!(f.auth.parts().count(), 0);
        assert!(f.auth.snapshot(Vec::new(), 1).contract.is_empty());
        assert_eq!(f.auth.mode(), Mode::Armed);
        f.run("policy", Command::Detach, 2).expect("detach");
        assert!(f.auth.binding(&pid("policy")).is_none());
    }

    #[test]
    fn estop_latches_and_needs_clear_disarm_arm() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let lease = f.grant("policy", "cmd1", 0).expect("lease");
        f.run(
            "stranger",
            Command::Estop {
                reason: "red button".into(),
                code: ReasonCode::EstopRemote,
            },
            1,
        )
        .expect("estop");
        assert_eq!(f.auth.mode(), Mode::Estop);
        assert!(f.auth.estop_latched());
        let epoch = f.auth.epoch();
        assert!(
            f.auth
                .check_fence(f.ch("cmd1"), lease.token, epoch, 2)
                .is_err()
        );
        let revision = f.auth.revision();
        f.run(
            "operator",
            Command::Estop {
                reason: "again".into(),
                code: ReasonCode::EstopLocal,
            },
            2,
        )
        .expect("estop");
        assert_eq!(f.auth.revision(), revision, "a second estop is a no-op");
        assert_eq!(
            Fixture::status(f.run("operator", Command::Arm, 3)),
            CtlStatus::EstopLatched
        );
        assert_eq!(
            Fixture::status(f.grant("operator", "cmd1", 3)),
            CtlStatus::EstopLatched
        );
        assert_eq!(
            Fixture::status(f.run("operator", Command::Clear, 3)),
            CtlStatus::Forbidden
        );
        f.run("robot", Command::Clear, 4).expect("local clear");
        assert_eq!(f.auth.mode_reason().code, ReasonCode::EstopCleared);
        // A local latch reported while cleared latches again, as the edge's.
        let local = Command::Estop {
            reason: "button".into(),
            code: ReasonCode::EstopLocal,
        };
        f.run("robot", local, 4).expect("relatch");
        assert!(f.auth.estop_latched());
        assert_eq!(f.auth.mode_reason().code, ReasonCode::EstopLocal);
        f.run("robot", Command::Clear, 4).expect("local clear");
        assert_eq!(
            Fixture::status(f.run("operator", Command::Arm, 5)),
            CtlStatus::InvalidTransition
        );
        f.run("operator", Command::Disarm, 6).expect("disarm");
        f.run("operator", Command::Arm, 7).expect("arm");
        assert_eq!(f.auth.epoch(), epoch.next());
    }

    #[test]
    fn an_attach_requiring_a_missing_feature_is_incompatible() {
        let mut f = Fixture::new();
        let future = |requires: Vec<String>| Command::Attach {
            name: "future".into(),
            roles: vec![Role::Controller],
            priority: PriorityClass::Model,
            protocol_minor: PROTOCOL_MINOR + 1,
            supports: vec!["merge.blend".into()],
            requires,
            workflow_edition: String::new(),
            workflow_editions: Vec::new(),
        };
        assert_eq!(
            Fixture::status(f.run("future", future(vec!["merge.blend".into()]), 0)),
            CtlStatus::Incompatible
        );
        assert!(!f.auth.is_attached(&pid("future")));
        f.run("future", future(Vec::new()), 0)
            .expect("a newer minor with no requirement attaches");
        let snap = f.auth.snapshot(Vec::new(), 0);
        assert_eq!(
            snap.participant(&pid("future")).map(|p| p.supports.clone()),
            Some(vec!["merge.blend".to_string()])
        );
    }

    #[test]
    fn the_snapshot_fence_matches_the_host() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let lease = f.grant("policy", "cmd3", 0).expect("lease");
        let snap = f.auth.snapshot(Vec::new(), 1);
        snap.check_fence(f.ch("cmd3"), lease.token, snap.epoch)
            .expect("fence");
        assert_eq!(snap.protocol_minor, PROTOCOL_MINOR);
        assert_eq!(snap.clock_mode, ClockMode::Wall);
        assert_eq!(
            snap.part_at(&pid("robot"), "cmd3").map(|p| p.channel),
            Some(f.ch("cmd3"))
        );
        assert_eq!(snap.group_members(&pid("robot"), "arm").count(), 2);
        let applied = snap
            .part(f.ch("cmd3"))
            .and_then(|p| p.applied_channel)
            .expect("applied");
        assert_eq!(snap.stream(applied).map(|s| s.applied), Some(true));
        assert!(snap.binding(&pid("policy")).is_some());
        let later = f.auth.snapshot(Vec::new(), 2);
        assert!(later.is_newer_than(&snap));
    }

    #[test]
    fn a_virtual_session_ticks_through_the_barrier() {
        let mut auth = Authority::new(
            SessionId::new(),
            AuthorityConfig {
                virtual_clock: Some(VirtualConfig::default()),
                ..AuthorityConfig::default()
            },
        );
        let mut f = {
            for (id, roles, priority) in [
                ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
                ("policy", vec![Role::Controller], PriorityClass::Model),
            ] {
                auth.apply(
                    Request {
                        from: pid(id),
                        epoch: None,
                        command: attach(id, roles, priority),
                    },
                    0,
                )
                .expect("attach");
            }
            Fixture { auth }
        };
        f.run("robot", declare(&robot_map("")), 0).expect("declare");
        let channels = vec![f.ch("cmd1")];
        f.run(
            "policy",
            Command::DeclareBinding {
                fingerprint: Fingerprint::from(5),
                revision: 1,
                table_json: "{}".into(),
                channels,
                replan_hz: 500.0,
                contracts: Vec::new(),
                grids: Vec::new(),
            },
            0,
        )
        .expect("bind at 500 Hz: every second 1 ms tick");
        f.run("policy", Command::Arm, 0).expect("arm");
        assert_eq!(
            f.auth.snapshot(Vec::new(), 0).clock_mode,
            ClockMode::Virtual
        );
        let ready = |tick| Command::Ready {
            tick,
            watermarks: Vec::new(),
        };
        f.run("robot", ready(0), 1).expect("ready");
        let ticks = f.auth.take_ticks();
        assert!(matches!(ticks.as_slice(), [VirtualTick::Observe(s)] if s.due == [pid("policy")]));
        let revision = f.auth.revision();
        f.run(
            "policy",
            Command::Decided {
                tick: 0,
                chunks: Vec::new(),
            },
            2,
        )
        .expect("decided");
        assert!(matches!(f.auth.take_ticks().as_slice(), [VirtualTick::Step(s)] if s.tick == 0));
        assert_eq!(f.auth.revision(), revision, "ticks are not structural");
        f.run("robot", ready(1), 3).expect("ready");
        assert_eq!(f.auth.take_ticks().len(), 2, "not due: observe and step");
        let stale = f.auth.apply(
            Request {
                from: pid("robot"),
                epoch: Some(Epoch(1)),
                command: ready(2),
            },
            4,
        );
        assert_eq!(Fixture::status(stale), CtlStatus::StaleEpoch);

        // A wall-clock session refuses tick verbs.
        let mut wall = Fixture::new();
        assert_eq!(
            Fixture::status(wall.run("robot", ready(0), 0)),
            CtlStatus::Unsupported
        );
    }
}
