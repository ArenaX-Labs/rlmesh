//! Authority state (design §2, §7): epoch, roster, mode, e-stop latch and
//! lease table, with the session host as its single writer.
//!
//! [`Authority`] is sans-IO: every method takes the host's session clock
//! (`now_ns`) and returns the events it caused, so the state machine is tested
//! without a network and the host only has to serialise calls into it.
//!
//! - Lease priority classes: `safety > operator > model > scripted`. A higher
//!   class preempts a lower one and gets a fresh token, which fences the old
//!   holder's in-flight chunks; an equal or lower class gets `LeaseBusy`.
//! - Leases have a TTL, renewed by every heartbeat from the holder on the
//!   current epoch; a lapse while ACTIVE takes the session SAFE.
//! - ESTOP is accepted from anyone (refusing an e-stop is never safer) and
//!   latches; only an embodiment's local clear releases the latch, and then a
//!   disarm and an arm (a fresh epoch) are needed.

use std::collections::BTreeMap;
use std::fmt;

use rlmesh_wire::authority::fence;
use rlmesh_wire::version::{PROTOCOL_MINOR, missing_capabilities};
use rlmesh_wire::{
    AuthoritySnapshot, ChannelId, ChannelSpec, ClockMapping, Command, CtlError, CtlStatus,
    Direction, Epoch, FenceError, Lease, LeaseToken, Mode, ParticipantId, ParticipantInfo,
    PriorityClass, Request, Role, SessionId,
};

/// Why a participant left the roster.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DetachReason {
    /// It sent `Detach`.
    Requested,
    /// Its liveliness token went away (crash, network loss).
    LivelinessLost,
}

/// Why a lease ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LeaseEnd {
    /// The holder released it (or detached).
    Released,
    /// The holder's heartbeats stopped for longer than the lease's ttl.
    Lapsed,
    /// The host took it back (disarm, e-stop, SAFE, channel removal).
    Revoked(&'static str),
}

/// Something the authority state did, for publication, logging and evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
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
        reason: String,
    },
    /// A lease was granted on a free channel.
    LeaseGranted {
        /// The lease.
        lease: Lease,
    },
    /// A higher class took a held lease.
    LeasePreempted {
        /// The fenced lease.
        previous: Lease,
        /// The new lease.
        by: Lease,
    },
    /// A lease ended.
    LeaseEnded {
        /// The lease.
        lease: Lease,
        /// Why.
        reason: LeaseEnd,
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
    /// Channels joined the table.
    ChannelsDeclared {
        /// Their owner.
        owner: ParticipantId,
        /// The new channels.
        channels: Vec<ChannelId>,
    },
    /// Channels left the table (their owner detached).
    ChannelsRemoved {
        /// Their owner.
        owner: ParticipantId,
        /// The removed channels.
        channels: Vec<ChannelId>,
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
            Self::LeaseGranted { lease } => write!(
                f,
                "lease {} granted to {} ({:?}) token {}",
                lease.channel, lease.holder, lease.priority, lease.token
            ),
            Self::LeasePreempted { previous, by } => write!(
                f,
                "lease {} preempted: {} ({:?}, {}) -> {} ({:?}, {})",
                by.channel,
                previous.holder,
                previous.priority,
                previous.token,
                by.holder,
                by.priority,
                by.token
            ),
            Self::LeaseEnded { lease, reason } => write!(
                f,
                "lease {} of {} token {} ended ({reason:?})",
                lease.channel, lease.holder, lease.token
            ),
            Self::EstopLatched { by, reason } => write!(f, "ESTOP latched by {by}: {reason}"),
            Self::EstopCleared { by } => write!(f, "ESTOP latch cleared by {by}"),
            Self::ChannelsDeclared { owner, channels } => {
                write!(f, "{owner} declared channels {channels:?}")
            }
            Self::ChannelsRemoved { owner, channels } => {
                write!(f, "channels {channels:?} of {owner} removed")
            }
        }
    }
}

/// The authority's rules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityConfig {
    /// Default lease ttl: how long the holder's heartbeats may stop before
    /// its lease lapses.
    pub lease_grace_ns: u64,
    /// The capabilities the session supports (see
    /// [`rlmesh_wire::version`]).
    pub capabilities: Vec<String>,
}

impl Default for AuthorityConfig {
    fn default() -> Self {
        Self {
            lease_grace_ns: 250_000_000,
            capabilities: rlmesh_wire::version::CAPABILITIES
                .iter()
                .map(ToString::to_string)
                .collect(),
        }
    }
}

/// What an accepted verb produced.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// The granted lease, for `GrantLease`.
    pub lease: Option<Lease>,
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
    roster: BTreeMap<ParticipantId, ParticipantInfo>,
    leases: BTreeMap<ChannelId, Lease>,
    channels: BTreeMap<ChannelId, ChannelSpec>,
    /// Channels removed in this epoch: an id is never re-declared with a
    /// different schema within one epoch, so an in-flight sample or chunk of
    /// the old stream can never be read with the new layout.
    retired: BTreeMap<ChannelId, ChannelSpec>,
    last_token: u64,
}

impl Authority {
    /// A fresh session: DISARMED in epoch 1.
    pub fn new(session: SessionId, config: AuthorityConfig) -> Self {
        Self {
            session,
            config,
            // Epoch 0 means "unchecked" on the wire, so sessions start at 1.
            epoch: Epoch::FIRST,
            revision: 0,
            mode: Mode::Disarmed,
            estop_latched: false,
            roster: BTreeMap::new(),
            leases: BTreeMap::new(),
            channels: BTreeMap::new(),
            retired: BTreeMap::new(),
            last_token: 0,
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

    /// The lease on `channel`.
    pub fn lease(&self, channel: ChannelId) -> Option<&Lease> {
        self.leases.get(&channel)
    }

    /// A declared channel.
    pub fn channel(&self, id: ChannelId) -> Option<&ChannelSpec> {
        self.channels.get(&id)
    }

    /// Every declared channel.
    pub fn channels(&self) -> impl Iterator<Item = &ChannelSpec> {
        self.channels.values()
    }

    /// Leases held.
    pub fn lease_count(&self) -> usize {
        self.leases.len()
    }

    /// Whether `id` is in the roster.
    pub fn is_attached(&self, id: &ParticipantId) -> bool {
        self.roster.contains_key(id)
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
            channels: self.channels.values().cloned().collect(),
            protocol_minor: PROTOCOL_MINOR,
            capabilities: self.config.capabilities.clone(),
        }
    }

    /// Applies a verb. Accepted verbs that change state bump the revision.
    pub fn apply(&mut self, request: Request, now_ns: i64) -> Result<Applied, CtlError> {
        let mut applied = Applied::default();
        let from = request.from;
        match request.command {
            Command::Attach {
                name,
                roles,
                priority,
                protocol_minor,
                capabilities,
            } => {
                if roles.is_empty() {
                    return Err(CtlError::new(
                        CtlStatus::InvalidRequest,
                        "attach declares no role",
                    ));
                }
                let missing = missing_capabilities(&capabilities, &self.config.capabilities);
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
                        capabilities,
                    },
                );
                applied.events.push(AuthorityEvent::Attached {
                    participant: from,
                    roles,
                    priority,
                });
            }
            Command::Detach => {
                self.require_attached(&from)?;
                self.end_leases_of(&from, LeaseEnd::Released, &mut applied.events);
                self.remove_channels_of(&from, &mut applied.events);
                self.roster.remove(&from);
                applied.events.push(AuthorityEvent::Detached {
                    participant: from,
                    reason: DetachReason::Requested,
                });
                self.settle_after_release("holder detached", &mut applied.events);
            }
            Command::Arm => {
                self.require_attached(&from)?;
                match self.mode {
                    Mode::Disarmed | Mode::Safe => {
                        self.epoch = self.epoch.next();
                        self.retired.clear();
                        self.set_mode(Mode::Armed, format!("armed by {from}"), &mut applied.events);
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
                        self.revoke_all("disarm", &mut applied.events);
                        self.set_mode(
                            Mode::Disarmed,
                            format!("disarmed by {from}"),
                            &mut applied.events,
                        );
                    }
                }
            }
            Command::GrantLease { channel, ttl_ns } => {
                let lease =
                    self.grant(&from, request.epoch, channel, ttl_ns, now_ns, &mut applied)?;
                applied.lease = Some(lease);
            }
            Command::ReleaseLease { channel, token } => {
                match self.leases.get(&channel) {
                    Some(lease) if lease.holder == from && lease.token == token => {}
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
                if let Some(lease) = self.leases.remove(&channel) {
                    applied.events.push(AuthorityEvent::LeaseEnded {
                        lease,
                        reason: LeaseEnd::Released,
                    });
                }
                self.settle_after_release("last lease released", &mut applied.events);
            }
            Command::Estop { reason } => {
                // Accepted from anyone: refusing an e-stop is never safer.
                // TODO(auth): restrict to authorised participants and log
                // unauthorised attempts as evidence.
                if self.mode != Mode::Estop || !self.estop_latched {
                    self.revoke_all("estop", &mut applied.events);
                    self.estop_latched = true;
                    applied.events.push(AuthorityEvent::EstopLatched {
                        by: from.clone(),
                        reason: reason.clone(),
                    });
                    self.set_mode(
                        Mode::Estop,
                        format!("estop by {from}: {reason}"),
                        &mut applied.events,
                    );
                }
            }
            Command::Clear => {
                let info = self.require_attached(&from)?;
                // "Local clear": only an embodiment may release its latch.
                // TODO(step 2): the edge runtime also needs its local
                // hardware clear before it sends this.
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
                applied
                    .events
                    .push(AuthorityEvent::EstopCleared { by: from });
            }
            Command::DeclareChannels(specs) => {
                self.declare(&from, specs, &mut applied.events)?;
            }
            Command::EnterSafe { reason } => {
                let info = self.require_attached(&from)?;
                if !info.has_role(Role::Embodiment) {
                    return Err(CtlError::new(
                        CtlStatus::Forbidden,
                        "only an embodiment's edge reports a fallback",
                    ));
                }
                // Outside ACTIVE there is nothing to stop: a no-op.
                self.enter_safe(&format!("{from} fell back: {reason}"), &mut applied.events);
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

    /// Lapses leases whose holder stopped heartbeating; ACTIVE goes SAFE.
    pub fn expire(&mut self, now_ns: i64) -> Vec<AuthorityEvent> {
        let mut events = Vec::new();
        let lapsed: Vec<ChannelId> = self
            .leases
            .values()
            .filter(|lease| now_ns > lease.expires_ns)
            .map(|lease| lease.channel)
            .collect();
        for channel in &lapsed {
            if let Some(lease) = self.leases.remove(channel) {
                events.push(AuthorityEvent::LeaseEnded {
                    lease,
                    reason: LeaseEnd::Lapsed,
                });
            }
        }
        if !lapsed.is_empty() {
            self.enter_safe("lease lapsed", &mut events);
            self.revision += 1;
        }
        events
    }

    /// The participant's liveliness token went away: it is gone, and any
    /// lease it held lapses.
    pub fn participant_lost(&mut self, id: &ParticipantId) -> Vec<AuthorityEvent> {
        let mut events = Vec::new();
        if self.roster.remove(id).is_none() {
            return events;
        }
        let before = events.len();
        self.end_leases_of(id, LeaseEnd::Lapsed, &mut events);
        let lapsed = events.len() > before;
        events.push(AuthorityEvent::Detached {
            participant: id.clone(),
            reason: DetachReason::LivelinessLost,
        });
        if lapsed {
            self.enter_safe("lease holder lost", &mut events);
        }
        self.revision += 1;
        events
    }

    /// All-or-nothing: either every channel is declared, or none is.
    fn declare(
        &mut self,
        from: &ParticipantId,
        specs: Vec<ChannelSpec>,
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<(), CtlError> {
        let info = self.require_attached(from)?;
        let is_embodiment = info.has_role(Role::Embodiment);
        let mut fresh = Vec::new();
        for mut spec in specs {
            spec.validate()?;
            if spec.direction == Direction::Actuate && !is_embodiment {
                return Err(CtlError::new(
                    CtlStatus::Forbidden,
                    format!(
                        "only an embodiment consumes actuation channels like {}",
                        spec.id
                    ),
                ));
            }
            if fresh.iter().any(|f: &ChannelSpec| f.id == spec.id) {
                return Err(CtlError::new(
                    CtlStatus::InvalidRequest,
                    format!("{} is declared twice", spec.id),
                ));
            }
            spec.owner = Some(from.clone());
            match self.channels.get(&spec.id) {
                Some(existing)
                    if existing.owner == spec.owner && existing.same_declaration(&spec) => {}
                Some(existing) => {
                    return Err(CtlError::new(
                        CtlStatus::ChannelConflict,
                        format!(
                            "{} is already declared as {} by {}",
                            spec.id,
                            existing.name,
                            existing.owner.as_ref().map_or("?", ParticipantId::as_str)
                        ),
                    ));
                }
                None => match self.retired.get(&spec.id) {
                    Some(old) if !old.same_declaration(&spec) => {
                        return Err(CtlError::new(
                            CtlStatus::ChannelConflict,
                            format!(
                                "{} was declared differently earlier in this epoch; \
                                 use a fresh id or re-arm",
                                spec.id
                            ),
                        ));
                    }
                    _ => fresh.push(spec),
                },
            }
        }
        if !fresh.is_empty() {
            events.push(AuthorityEvent::ChannelsDeclared {
                owner: from.clone(),
                channels: fresh.iter().map(|c| c.id).collect(),
            });
            for spec in fresh {
                self.channels.insert(spec.id, spec);
            }
        }
        Ok(())
    }

    /// A detaching owner takes its channels, and the leases on them, along.
    /// Liveliness loss keeps them: a reconnecting embodiment re-attaches to
    /// the same ids (its leases lapse as usual).
    fn remove_channels_of(&mut self, owner: &ParticipantId, events: &mut Vec<AuthorityEvent>) {
        let ids: Vec<ChannelId> = self
            .channels
            .values()
            .filter(|c| c.owner.as_ref() == Some(owner))
            .map(|c| c.id)
            .collect();
        if ids.is_empty() {
            return;
        }
        for id in &ids {
            if let Some(spec) = self.channels.remove(id) {
                self.retired.insert(*id, spec);
            }
            if let Some(lease) = self.leases.remove(id) {
                events.push(AuthorityEvent::LeaseEnded {
                    lease,
                    reason: LeaseEnd::Revoked("channel removed"),
                });
            }
        }
        events.push(AuthorityEvent::ChannelsRemoved {
            owner: owner.clone(),
            channels: ids,
        });
    }

    fn require_attached(&self, id: &ParticipantId) -> Result<&ParticipantInfo, CtlError> {
        self.roster
            .get(id)
            .ok_or_else(|| CtlError::new(CtlStatus::NotAttached, format!("{id} is not attached")))
    }

    fn grant(
        &mut self,
        from: &ParticipantId,
        epoch: Option<Epoch>,
        channel: ChannelId,
        ttl_ns: Option<u64>,
        now_ns: i64,
        applied: &mut Applied,
    ) -> Result<Lease, CtlError> {
        let info = self.require_attached(from)?;
        if !info.has_role(Role::Controller) {
            return Err(CtlError::new(
                CtlStatus::Forbidden,
                format!("{from} is not a controller"),
            ));
        }
        let priority = info.priority;
        match self.channels.get(&channel) {
            Some(spec) if spec.direction == Direction::Actuate => {}
            Some(_) => {
                return Err(CtlError::new(
                    CtlStatus::UnknownChannel,
                    format!("{channel} is not an actuation channel"),
                ));
            }
            None => {
                return Err(CtlError::new(
                    CtlStatus::UnknownChannel,
                    format!("{channel} is not declared"),
                ));
            }
        }
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
        let ttl_ns = ttl_ns
            .filter(|ttl| *ttl > 0)
            .unwrap_or(self.config.lease_grace_ns);
        let expires_ns = now_ns.saturating_add_unsigned(ttl_ns);

        let previous = match self.leases.get_mut(&channel) {
            Some(lease) if &lease.holder == from => {
                // Re-grant to the holder: a renewal under the same token.
                lease.ttl_ns = ttl_ns;
                lease.expires_ns = expires_ns;
                return Ok(lease.clone());
            }
            Some(lease) if lease.priority < priority => Some(lease.clone()),
            Some(lease) => {
                return Err(CtlError::new(
                    CtlStatus::LeaseBusy,
                    format!(
                        "{channel} is held by {} at {:?}; {from} at {priority:?} cannot preempt it",
                        lease.holder, lease.priority
                    ),
                ));
            }
            None => None,
        };

        self.last_token += 1;
        let lease = Lease {
            channel,
            holder: from.clone(),
            priority,
            token: LeaseToken(self.last_token),
            ttl_ns,
            granted_ns: now_ns,
            expires_ns,
        };
        self.leases.insert(channel, lease.clone());
        applied.events.push(match previous {
            Some(previous) => AuthorityEvent::LeasePreempted {
                previous,
                by: lease.clone(),
            },
            None => AuthorityEvent::LeaseGranted {
                lease: lease.clone(),
            },
        });
        if self.mode == Mode::Armed {
            self.set_mode(
                Mode::Active,
                format!("lease granted to {from}"),
                &mut applied.events,
            );
        }
        Ok(lease)
    }

    fn set_mode(&mut self, to: Mode, reason: String, events: &mut Vec<AuthorityEvent>) {
        if self.mode == to {
            return;
        }
        events.push(AuthorityEvent::ModeChanged {
            from: self.mode,
            to,
            epoch: self.epoch,
            reason,
        });
        self.mode = to;
    }

    fn end_leases_of(
        &mut self,
        id: &ParticipantId,
        reason: LeaseEnd,
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
                events.push(AuthorityEvent::LeaseEnded {
                    lease,
                    reason: reason.clone(),
                });
            }
        }
    }

    fn revoke_all(&mut self, why: &'static str, events: &mut Vec<AuthorityEvent>) {
        for (_, lease) in std::mem::take(&mut self.leases) {
            events.push(AuthorityEvent::LeaseEnded {
                lease,
                reason: LeaseEnd::Revoked(why),
            });
        }
    }

    /// ACTIVE with no lease left drops back to ARMED.
    fn settle_after_release(&mut self, reason: &str, events: &mut Vec<AuthorityEvent>) {
        if self.mode == Mode::Active && self.leases.is_empty() {
            self.set_mode(Mode::Armed, reason.to_string(), events);
        }
    }

    /// A lapse while ACTIVE takes the whole session to SAFE and revokes every
    /// other lease: re-arming (a fresh epoch) is the way back.
    fn enter_safe(&mut self, reason: &str, events: &mut Vec<AuthorityEvent>) {
        if self.mode == Mode::Active {
            self.revoke_all("safe", events);
            self.set_mode(Mode::Safe, reason.to_string(), events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_wire::{ChannelPriority, ChannelQos, TensorLayout};

    const GRACE: u64 = 100;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    struct Fixture {
        auth: Authority,
    }

    impl Fixture {
        /// robot (embodiment), policy (model), operator, script, viewer.
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
                        command: Command::Attach {
                            name: id.into(),
                            roles,
                            priority,
                            protocol_minor: PROTOCOL_MINOR,
                            capabilities: Vec::new(),
                        },
                    },
                    0,
                )
                .expect("attach");
            }
            let channels = (1..=9)
                .map(|id| {
                    ChannelSpec::new(
                        ChannelId(id),
                        format!("cmd{id}"),
                        Direction::Actuate,
                        "action/joint_pos",
                        vec![TensorLayout::f32(&[6])],
                        500.0,
                        ChannelQos::realtime(ChannelPriority::Control),
                    )
                })
                .collect();
            auth.apply(
                Request {
                    from: pid("robot"),
                    epoch: None,
                    command: Command::DeclareChannels(channels),
                },
                0,
            )
            .expect("declare");
            Self { auth }
        }

        fn run(&mut self, from: &str, command: Command, now: i64) -> Result<Applied, CtlError> {
            self.auth.apply(
                Request {
                    from: pid(from),
                    epoch: None,
                    command,
                },
                now,
            )
        }

        fn grant(&mut self, from: &str, channel: u32, now: i64) -> Result<Lease, CtlError> {
            let epoch = Some(self.auth.epoch());
            self.auth
                .apply(
                    Request {
                        from: pid(from),
                        epoch,
                        command: Command::GrantLease {
                            channel: ChannelId(channel),
                            ttl_ns: None,
                        },
                    },
                    now,
                )
                .map(|applied| applied.lease.expect("lease"))
        }

        fn status(result: Result<impl fmt::Debug, CtlError>) -> CtlStatus {
            result.expect_err("refused").status
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
    fn first_lease_activates_and_releasing_the_last_rearms() {
        let mut f = Fixture::new();
        assert_eq!(
            Fixture::status(f.grant("policy", 7, 0)),
            CtlStatus::InvalidTransition,
            "grant before arm"
        );
        f.run("operator", Command::Arm, 0).expect("arm");
        assert_eq!(
            Fixture::status(f.grant("viewer", 7, 0)),
            CtlStatus::Forbidden
        );
        let lease = f.grant("policy", 7, 10).expect("grant");
        assert_eq!(f.auth.mode(), Mode::Active);
        assert_eq!(lease.expires_ns, 10 + GRACE as i64);
        f.auth
            .check_fence(ChannelId(7), lease.token, f.auth.epoch(), 20)
            .expect("fence");

        let revision = f.auth.revision();
        let renewed = f.grant("policy", 7, 50).expect("re-grant");
        assert_eq!(
            renewed.token, lease.token,
            "re-grant renews under the same token"
        );
        assert_eq!(
            f.auth.revision(),
            revision,
            "renewal is not a structural change"
        );

        f.run(
            "policy",
            Command::ReleaseLease {
                channel: ChannelId(7),
                token: lease.token,
            },
            60,
        )
        .expect("release");
        assert_eq!(f.auth.mode(), Mode::Armed);
        assert!(f.auth.lease(ChannelId(7)).is_none());
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
                    channel: ChannelId(1),
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
        let model = f.grant("policy", 1, 0).expect("model lease");
        let applied = f
            .auth
            .apply(
                Request {
                    from: pid("operator"),
                    epoch: Some(f.auth.epoch()),
                    command: Command::GrantLease {
                        channel: ChannelId(1),
                        ttl_ns: None,
                    },
                },
                5,
            )
            .expect("preempt");
        let operator = applied.lease.clone().expect("lease");
        assert!(operator.token > model.token, "preemption bumps the token");
        assert!(matches!(
            applied.events.as_slice(),
            [AuthorityEvent::LeasePreempted { previous, by }]
                if previous.holder == pid("policy") && by.holder == pid("operator")
        ));
        assert_eq!(f.auth.mode(), Mode::Active);

        let epoch = f.auth.epoch();
        assert_eq!(
            f.auth.check_fence(ChannelId(1), model.token, epoch, 6),
            Err(FenceError::StaleToken {
                channel: ChannelId(1),
                current: operator.token,
                got: model.token
            })
        );
        f.auth
            .check_fence(ChannelId(1), operator.token, epoch, 6)
            .expect("new token");
        assert_eq!(
            Fixture::status(f.run(
                "policy",
                Command::ReleaseLease {
                    channel: ChannelId(1),
                    token: model.token
                },
                7
            )),
            CtlStatus::StaleToken
        );
        // The model cannot take it back, nor can an equal class steal it.
        assert_eq!(
            Fixture::status(f.grant("policy", 1, 8)),
            CtlStatus::LeaseBusy
        );
        // After the operator releases, the model may hold it again.
        f.run(
            "operator",
            Command::ReleaseLease {
                channel: ChannelId(1),
                token: operator.token,
            },
            9,
        )
        .expect("release");
        let again = f.grant("policy", 1, 10).expect("model again");
        assert!(again.token > operator.token);
    }

    #[test]
    fn equal_or_lower_classes_cannot_preempt() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", 1, 0).expect("lease");
        assert_eq!(
            Fixture::status(f.grant("script", 1, 1)),
            CtlStatus::LeaseBusy
        );
        // Other channels are independent.
        f.grant("script", 2, 1).expect("other channel");
    }

    #[test]
    fn an_embodiments_fallback_takes_the_session_safe() {
        let mut f = Fixture::new();
        let safe = |reason: &str| Command::EnterSafe {
            reason: reason.into(),
        };
        // Outside ACTIVE: nothing to stop.
        f.run("robot", safe("idle"), 0).expect("no-op");
        assert_eq!(f.auth.mode(), Mode::Disarmed);
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", 1, 0).expect("lease");
        assert_eq!(
            Fixture::status(f.run("policy", safe("not mine"), 1)),
            CtlStatus::Forbidden
        );
        f.run("robot", safe("queue dry past hold_max"), 2)
            .expect("enter safe");
        assert_eq!(f.auth.mode(), Mode::Safe);
        assert!(f.auth.lease(ChannelId(1)).is_none(), "SAFE revokes leases");
    }

    #[test]
    fn heartbeats_renew_and_their_absence_lapses_to_safe() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let a = f.grant("policy", 1, 0).expect("lease");
        f.grant("script", 2, 0).expect("lease");
        let epoch = f.auth.epoch();
        for now in [50, 100, 150] {
            f.auth.heartbeat(&pid("policy"), epoch, now);
            f.auth.heartbeat(&pid("script"), epoch, now);
            assert!(f.auth.expire(now).is_empty());
        }
        // A heartbeat from an older epoch renews nothing.
        f.auth.heartbeat(&pid("policy"), Epoch(1), 200);
        f.auth.heartbeat(&pid("script"), epoch, 200);
        let events = f.auth.expire(251);
        assert_eq!(f.auth.mode(), Mode::Safe);
        assert!(events.iter().any(|e| matches!(
            e,
            AuthorityEvent::LeaseEnded { lease, reason: LeaseEnd::Lapsed } if lease.token == a.token
        )));
        assert!(
            events.iter().any(|e| matches!(
                e,
                AuthorityEvent::LeaseEnded {
                    reason: LeaseEnd::Revoked("safe"),
                    ..
                }
            )),
            "SAFE revokes the other leases"
        );
        assert!(f.auth.lease(ChannelId(2)).is_none());
        assert_eq!(
            Fixture::status(f.grant("policy", 1, 260)),
            CtlStatus::InvalidTransition
        );
        f.run("operator", Command::Arm, 270)
            .expect("re-arm from SAFE");
        assert_eq!(f.auth.epoch(), epoch.next());
        assert_eq!(
            f.auth.check_fence(ChannelId(1), a.token, epoch, 270),
            Err(FenceError::StaleEpoch {
                current: epoch.next(),
                got: epoch
            })
        );
    }

    #[test]
    fn losing_the_holder_goes_safe_and_losing_an_observer_does_not() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", 1, 0).expect("lease");
        f.auth.participant_lost(&pid("viewer"));
        assert_eq!(f.auth.mode(), Mode::Active);
        f.auth.participant_lost(&pid("policy"));
        assert_eq!(f.auth.mode(), Mode::Safe);
        assert!(!f.auth.is_attached(&pid("policy")));
    }

    #[test]
    fn a_detaching_holder_releases_back_to_armed() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        f.grant("policy", 1, 0).expect("lease");
        f.run("policy", Command::Detach, 1).expect("detach");
        assert_eq!(f.auth.mode(), Mode::Armed);
    }

    #[test]
    fn estop_latches_and_needs_clear_disarm_arm() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let lease = f.grant("policy", 1, 0).expect("lease");
        // Accepted even from a participant that never attached.
        f.run(
            "stranger",
            Command::Estop {
                reason: "red button".into(),
            },
            1,
        )
        .expect("estop");
        assert_eq!(f.auth.mode(), Mode::Estop);
        assert!(f.auth.estop_latched());
        assert!(f.auth.lease(ChannelId(1)).is_none());
        let epoch = f.auth.epoch();
        assert!(
            f.auth
                .check_fence(ChannelId(1), lease.token, epoch, 2)
                .is_err()
        );

        let revision = f.auth.revision();
        f.run(
            "operator",
            Command::Estop {
                reason: "again".into(),
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
            Fixture::status(f.run("operator", Command::Disarm, 3)),
            CtlStatus::EstopLatched
        );
        assert_eq!(
            Fixture::status(f.grant("operator", 1, 3)),
            CtlStatus::EstopLatched
        );
        assert_eq!(
            Fixture::status(f.run("operator", Command::Clear, 3)),
            CtlStatus::Forbidden
        );

        f.run("robot", Command::Clear, 4).expect("local clear");
        assert_eq!(
            f.auth.mode(),
            Mode::Estop,
            "clear alone does not leave ESTOP"
        );
        assert_eq!(
            Fixture::status(f.run("operator", Command::Arm, 5)),
            CtlStatus::InvalidTransition
        );
        f.run("operator", Command::Disarm, 6).expect("disarm");
        f.run("operator", Command::Arm, 7).expect("arm");
        assert_eq!(f.auth.mode(), Mode::Armed);
        assert_eq!(f.auth.epoch(), epoch.next());
    }

    #[test]
    fn tokens_are_unique_and_increasing_across_channels() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let tokens: Vec<LeaseToken> = (1..5)
            .map(|ch| f.grant("policy", ch, 0).expect("lease").token)
            .collect();
        assert!(tokens.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn channels_need_a_consistent_owner_and_leases_need_actuation() {
        let mut f = Fixture::new();
        let sense = ChannelSpec::new(
            ChannelId(20),
            "ft",
            Direction::Sense,
            "proprio/eef_wrench",
            vec![TensorLayout::f32(&[6])],
            1000.0,
            ChannelQos::realtime(ChannelPriority::Realtime),
        );
        let applied = f
            .run("robot", Command::DeclareChannels(vec![sense.clone()]), 0)
            .expect("declare");
        assert_eq!(applied.events.len(), 1);
        let applied = f
            .run("robot", Command::DeclareChannels(vec![sense.clone()]), 0)
            .expect("redeclare");
        assert!(applied.events.is_empty(), "identical re-declaration");
        let mut changed = sense.clone();
        changed.rate.nominal_hz = 500.0;
        assert_eq!(
            Fixture::status(f.run("robot", Command::DeclareChannels(vec![changed]), 0)),
            CtlStatus::ChannelConflict
        );
        assert_eq!(
            Fixture::status(f.run("viewer", Command::DeclareChannels(vec![sense]), 0)),
            CtlStatus::ChannelConflict,
            "another owner"
        );
        let mut actuate = f.auth.channel(ChannelId(1)).expect("cmd1").clone();
        actuate.id = ChannelId(21);
        assert_eq!(
            Fixture::status(f.run("viewer", Command::DeclareChannels(vec![actuate]), 0)),
            CtlStatus::Forbidden,
            "only an embodiment consumes actuation"
        );

        f.run("operator", Command::Arm, 0).expect("arm");
        for channel in [20, 99] {
            assert_eq!(
                Fixture::status(f.grant("policy", channel, 0)),
                CtlStatus::UnknownChannel
            );
        }
        f.grant("policy", 1, 0)
            .expect("lease on an actuation channel");

        // The owner detaching takes its channels and their leases along.
        let applied = f.run("robot", Command::Detach, 0).expect("detach");
        assert!(applied.events.iter().any(|e| matches!(
            e,
            AuthorityEvent::LeaseEnded {
                reason: LeaseEnd::Revoked("channel removed"),
                ..
            }
        )));
        assert!(f.auth.snapshot(Vec::new(), 0).channels.is_empty());
        assert_eq!(f.auth.mode(), Mode::Armed);
    }

    #[test]
    fn the_snapshot_fence_matches_the_host() {
        let mut f = Fixture::new();
        f.run("operator", Command::Arm, 0).expect("arm");
        let lease = f.grant("policy", 3, 0).expect("lease");
        let snap = f.auth.snapshot(Vec::new(), 1);
        snap.check_fence(ChannelId(3), lease.token, snap.epoch)
            .expect("fence");
        assert_eq!(snap.protocol_minor, PROTOCOL_MINOR);
        assert_eq!(
            snap.check_fence(ChannelId(4), lease.token, snap.epoch),
            Err(FenceError::NoLease {
                channel: ChannelId(4)
            })
        );
        let later = f.auth.snapshot(Vec::new(), 2);
        assert!(later.is_newer_than(&snap));
    }

    #[test]
    fn an_attach_requiring_a_missing_capability_is_incompatible() {
        let mut f = Fixture::new();
        let attach = |capabilities: Vec<String>| Command::Attach {
            name: "future".into(),
            roles: vec![Role::Controller],
            priority: PriorityClass::Model,
            protocol_minor: PROTOCOL_MINOR + 1,
            capabilities,
        };
        assert_eq!(
            Fixture::status(f.run("future", attach(vec!["merge.blend".into()]), 0)),
            CtlStatus::Incompatible
        );
        assert!(!f.auth.is_attached(&pid("future")));
        f.run("future", attach(Vec::new()), 0)
            .expect("a newer minor with no requirement attaches");
        let snap = f.auth.snapshot(Vec::new(), 0);
        assert_eq!(
            snap.participant(&pid("future")).map(|p| p.protocol_minor),
            Some(PROTOCOL_MINOR + 1)
        );
    }

    #[test]
    fn a_removed_channel_id_is_not_reused_with_another_schema_in_the_epoch() {
        let mut f = Fixture::new();
        let mut spec = f.auth.channel(ChannelId(1)).expect("cmd1").clone();
        spec.owner = None;
        f.run("robot", Command::Detach, 0).expect("detach");
        f.run(
            "robot",
            Command::Attach {
                name: "robot".into(),
                roles: vec![Role::Embodiment],
                priority: PriorityClass::Scripted,
                protocol_minor: PROTOCOL_MINOR,
                capabilities: Vec::new(),
            },
            1,
        )
        .expect("re-attach");
        let mut changed = spec.clone();
        changed.layout = vec![TensorLayout::f32(&[7])];
        assert_eq!(
            Fixture::status(f.run("robot", Command::DeclareChannels(vec![changed.clone()]), 1)),
            CtlStatus::ChannelConflict
        );
        f.run("robot", Command::DeclareChannels(vec![spec]), 1)
            .expect("the same schema comes back");
        f.run("robot", Command::Detach, 2).expect("detach");
        f.run(
            "robot",
            Command::Attach {
                name: "robot".into(),
                roles: vec![Role::Embodiment],
                priority: PriorityClass::Scripted,
                protocol_minor: PROTOCOL_MINOR,
                capabilities: Vec::new(),
            },
            3,
        )
        .expect("re-attach");
        f.run("operator", Command::Arm, 3)
            .expect("arm: a fresh epoch");
        f.run("robot", Command::DeclareChannels(vec![changed]), 4)
            .expect("a new schema in a new epoch");
    }
}
