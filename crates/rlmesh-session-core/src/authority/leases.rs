//! The lease table: tokens, all-or-none grants with preemption and
//! command groups, releases and the ring of lease endings.

use alloc::collections::BTreeSet;

use super::{Authority, AuthorityEvent, ENDED, Grant};
use crate::prelude::*;
use crate::wire::authority::held_refusal;
use crate::wire::{
    ChannelId, CtlError, CtlStatus, Epoch, Lease, LeaseEnd, LeaseToken, Mode, ParticipantId,
    Reason, ReasonCode, Role,
};

impl Authority {
    pub(super) fn next_token(&mut self) -> LeaseToken {
        if let Some(token) = self.minting.take() {
            return token;
        }
        self.last_token += 1;
        LeaseToken(self.last_token)
    }

    /// Reserves the next lease token, for a verb whose keys go out before
    /// it is applied ([`Self::apply_minting`]).
    pub fn reserve_token(&mut self) -> LeaseToken {
        self.last_token += 1;
        LeaseToken(self.last_token)
    }

    /// Validates the whole request, then grants: all or none.
    pub(super) fn grant(
        &mut self,
        from: &ParticipantId,
        device: &str,
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
                return Err(invalid(format!(
                    "{} is named twice",
                    self.part_name(*channel)
                )));
            }
        }
        for channel in &grant.channels {
            let Some(part) = self.commandable(*channel) else {
                return Err(invalid(format!(
                    "{} is not a commandable part",
                    self.part_name(*channel)
                )));
            };
            if !self.is_attached(&part.owner) {
                return Err(CtlError::new(
                    CtlStatus::NotReady,
                    format!(
                        "{} is not present: its parts take a lease once it is back",
                        part.owner
                    ),
                ));
            }
            if let Some(why) = self.held.get(&part.owner) {
                return Err(CtlError::new(
                    CtlStatus::NotReady,
                    held_refusal(&part.owner, why, self.scene_run()),
                ));
            }
            // Before pre-emption and busy checks: a device pre-empts only
            // with a class it is authorized for on that part.
            self.config
                .access
                .check_lease(device, &part.owner, part.path.as_str(), priority)?;
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
                format!(
                    "{} is outside binding {}",
                    self.part_name(*outside),
                    record.fingerprint
                ),
            ));
        }
        // Install before authority.
        for c in &record.contracts {
            let needed = self.needed.get(&(from.clone(), c.channel)).copied();
            let installed = self
                .demands
                .get(&c.channel)
                .is_some_and(|d| needed.is_some_and(|n| d.installed(n)));
            if !installed {
                let path = self.part_name(c.channel);
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
                return Err(invalid(format!(
                    "{} is not held by {from}",
                    self.part_name(*channel)
                )));
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
                            "{} is held by {} at {:?}; {from} at {priority:?} cannot preempt it",
                            self.part_name(*channel),
                            l.holder,
                            l.priority
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
                                "{from} holds {} in the same command group: release it in the same request",
                                self.part_name(peer)
                            ),
                        ));
                    }
                    Some(l) if l.priority < priority => preempted.push((peer, *channel)),
                    Some(l) => {
                        return Err(CtlError::new(
                            CtlStatus::GroupBusy,
                            format!(
                                "{}, in the command group of {}, is held by {} at {:?}",
                                self.part_name(peer),
                                self.part_name(*channel),
                                l.holder,
                                l.priority
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
                key_id: 0,
            };
            let part = self.part_name(*channel);
            events.push(match self.leases.insert(*channel, lease.clone()) {
                Some(previous) => {
                    let why = self.reason(ReasonCode::Preempted, from, "");
                    self.record_end(&previous, why, Some(&lease));
                    AuthorityEvent::LeasePreempted {
                        previous,
                        part: part.clone(),
                        by: lease.clone(),
                        by_part: part,
                    }
                }
                None => AuthorityEvent::LeaseGranted {
                    lease: lease.clone(),
                    part,
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
                let (part, by_part) = (self.part_name(peer), self.part_name(by.channel));
                events.push(AuthorityEvent::LeasePreempted {
                    previous,
                    part,
                    by,
                    by_part,
                });
            }
        }
        if self.mode == Mode::Armed && !self.leases.is_empty() {
            let why = self.reason(ReasonCode::FirstLease, from, "");
            self.set_mode(Mode::Active, why, events);
        }
        Ok(granted)
    }

    pub(super) fn release(
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
                            "{} is held by {} with {}, not {from} with {token}",
                            self.part_name(*channel),
                            lease.holder,
                            lease.token
                        ),
                    ));
                }
                None => {
                    return Err(CtlError::new(
                        CtlStatus::StaleToken,
                        format!("no lease on {}", self.part_name(*channel)),
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

    /// Keeps a lease ending in the snapshot's ring.
    pub(super) fn record_end(&mut self, lease: &Lease, reason: Reason, by: Option<&Lease>) {
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
    pub(super) fn end(&mut self, lease: Lease, reason: Reason, events: &mut Vec<AuthorityEvent>) {
        self.record_end(&lease, reason.clone(), None);
        let part = self.part_name(lease.channel);
        events.push(AuthorityEvent::LeaseEnded {
            lease,
            part,
            reason,
        });
    }

    pub(super) fn end_leases_of(
        &mut self,
        id: &ParticipantId,
        reason: &Reason,
        events: &mut Vec<AuthorityEvent>,
    ) {
        for channel in self.held_by(id) {
            if let Some(lease) = self.leases.remove(&channel) {
                self.end(lease, reason.clone(), events);
            }
        }
    }

    /// `holder` went away: the leases on `channels` end with `why`, and
    /// their edges park the parts as after a release. Says whether any did.
    pub(super) fn release_holder(
        &mut self,
        holder: &ParticipantId,
        channels: &[ChannelId],
        why: &Reason,
        events: &mut Vec<AuthorityEvent>,
    ) -> bool {
        let mut parts = Vec::with_capacity(channels.len());
        for channel in channels {
            if let Some(lease) = self.leases.remove(channel) {
                if let Some(part) = self.parts.get(channel) {
                    parts.push(format!("{}:{}", part.owner, part.path.as_str()));
                }
                self.end(lease, why.clone(), events);
            }
        }
        let released = !parts.is_empty();
        if released {
            events.push(AuthorityEvent::HolderLost {
                holder: holder.clone(),
                code: why.code,
                parts,
            });
        }
        released
    }

    /// The parts `holder` holds a lease on.
    pub(super) fn held_by(&self, holder: &ParticipantId) -> Vec<ChannelId> {
        self.leases
            .values()
            .filter(|lease| &lease.holder == holder)
            .map(|lease| lease.channel)
            .collect()
    }

    /// Ends every lease with `code`, carrying who and why from `cause`.
    pub(super) fn revoke_all(
        &mut self,
        code: ReasonCode,
        cause: &Reason,
        events: &mut Vec<AuthorityEvent>,
    ) {
        for (_, lease) in core::mem::take(&mut self.leases) {
            let why = Reason {
                code,
                ..cause.clone()
            };
            self.end(lease, why, events);
        }
    }
}
