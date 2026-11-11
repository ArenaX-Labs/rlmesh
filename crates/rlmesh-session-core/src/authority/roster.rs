//! Who is in the session: attach generations, presence by heartbeat,
//! loss, detach and access changes.

use alloc::collections::BTreeMap;

use super::{Authority, AuthorityEvent, DetachReason, DeviceAccess};
use crate::prelude::*;
use crate::wire::{
    ChannelId, CtlError, CtlStatus, Epoch, Mode, ParticipantId, ParticipantInfo, ReasonCode, Role,
};

impl Authority {
    /// A heartbeat from `from` arrived: it is present, and its leases renew
    /// when it is on the current epoch. Neither is a structural change.
    pub fn heartbeat(&mut self, from: &ParticipantId, epoch: Epoch, now_ns: i64) {
        if let Some(seen) = self.seen.get_mut(from) {
            seen.last_ns = now_ns;
            seen.hint_ns = None;
        }
        if epoch != self.epoch {
            return;
        }
        for lease in self.leases.values_mut().filter(|l| &l.holder == from) {
            lease.expires_ns = now_ns.saturating_add_unsigned(lease.ttl_ns);
        }
    }

    /// A participant's liveliness token went away: only a hint (anyone on
    /// the network can declare or drop a token). It is lost unless a
    /// heartbeat arrives within `lease_grace`.
    pub fn presence_hint(&mut self, id: &ParticipantId, now_ns: i64) {
        if let Some(seen) = self.seen.get_mut(id) {
            seen.hint_ns.get_or_insert(now_ns);
        }
    }

    /// Lapses leases whose holder stopped heartbeating (only those parts
    /// stop), loses participants whose heartbeats lapsed, and ends a virtual
    /// run whose barrier stalled.
    pub fn expire(&mut self, now_ns: i64) -> Vec<AuthorityEvent> {
        self.now_ns = now_ns;
        let mut events = Vec::new();
        let mut lapsed: BTreeMap<ParticipantId, Vec<ChannelId>> = BTreeMap::new();
        for lease in self.leases.values().filter(|l| now_ns > l.expires_ns) {
            lapsed
                .entry(lease.holder.clone())
                .or_default()
                .push(lease.channel);
        }
        let mut stops_run = None;
        for (holder, channels) in &lapsed {
            let why = self.reason(ReasonCode::HeartbeatLapse, holder, "");
            self.release_holder(holder, channels, &why, &mut events);
            if self.scene_member(holder) {
                stops_run.get_or_insert(why);
            }
        }
        match (stops_run, lapsed.keys().next()) {
            (Some(why), _) => self.enter_safe(why, &mut events),
            (None, Some(holder)) => self.settle_after_release(holder, "holder lapsed", &mut events),
            (None, None) => {}
        }
        // Never shorter than the lease grace, which a holder may count on.
        let hint = i64::try_from(self.config.lease_grace_ns).unwrap_or(i64::MAX);
        let presence = i64::try_from(self.config.presence_grace_ns)
            .unwrap_or(i64::MAX)
            .max(hint);
        let lost: Vec<ParticipantId> = self
            .seen
            .iter()
            .filter(|(_, s)| {
                now_ns.saturating_sub(s.last_ns) > presence
                    || s.hint_ns.is_some_and(|h| now_ns.saturating_sub(h) > hint)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in &lost {
            self.lose(id, &mut events);
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

    /// The participant's heartbeats lapsed. Any lease it held ends
    /// (`HOLDER_LOST`); a robot stays in the roster, absent, and the leases
    /// on its parts end with `ROBOT_LOST`. Everyone else goes on.
    pub fn participant_lost(&mut self, id: &ParticipantId, now_ns: i64) -> Vec<AuthorityEvent> {
        self.now_ns = now_ns;
        let mut events = Vec::new();
        self.lose(id, &mut events);
        if !events.is_empty() {
            self.revision += 1;
        }
        events
    }

    /// A participant is gone: any lease it held ends (`HOLDER_LOST`), and
    /// the session and every other holder go on. A robot stays in the
    /// roster, absent, while its parts remain: the leases on them end with
    /// `ROBOT_LOST`. Anyone else leaves the roster. Parts and bindings stay
    /// for a reconnect. Only a scene's run stops for a lost member: it waits
    /// for it, SAFE, until an explicit arm.
    pub(super) fn lose(&mut self, id: &ParticipantId, events: &mut Vec<AuthorityEvent>) {
        if !self.is_attached(id) {
            return;
        }
        let in_scene = self.scene_lost(id);
        self.seen.remove(id);
        let held = self.reason(ReasonCode::HolderLost, id, "");
        let released = self.release_holder(id, &self.held_by(id), &held, events);
        let robot = self.parts.values().any(|p| &p.owner == id);
        events.push(AuthorityEvent::Detached {
            participant: id.clone(),
            reason: DetachReason::Lost,
        });
        if robot {
            let why = self.reason(ReasonCode::RobotLost, id, "");
            let commanded: Vec<ChannelId> = self
                .parts
                .values()
                .filter(|p| &p.owner == id)
                .map(|p| p.channel)
                .collect();
            for channel in commanded {
                if let Some(lease) = self.leases.remove(&channel) {
                    self.end(lease, why.clone(), events);
                }
            }
            if let Some(info) = self.roster.get_mut(id) {
                info.present = false;
                info.left = Some(why.clone());
            }
            self.settle_after_release(id, "its robot was lost", events);
        } else {
            self.roster.remove(id);
            if in_scene && released {
                self.enter_safe(held, events);
            } else {
                self.settle_after_release(id, "holder lost", events);
            }
        }
        // In a scene's run an ARMED session goes SAFE too: once the member is back,
        // leases would otherwise take it
        // ACTIVE with no explicit arm.
        if in_scene && self.mode == Mode::Armed {
            let code = if robot {
                ReasonCode::RobotLost
            } else {
                ReasonCode::HolderLost
            };
            let why = self.reason(code, id, "");
            self.set_mode(Mode::Safe, why, events);
        }
        // Lost from a stopped session, nothing was driving it: the arm
        // that starts the session again is all it needs.
        if robot && matches!(self.mode, Mode::Armed | Mode::Active) {
            let why = self.reason(ReasonCode::RobotLost, id, "");
            self.hold(id, why, events);
        }
        self.needed.retain(|(p, _), _| p != id);
        self.recompute_demands(events);
        self.barrier_gone(id, self.now_ns);
    }

    /// `id` leaves the roster with its parts, binding and leases (`code`).
    pub(super) fn detach(
        &mut self,
        id: &ParticipantId,
        code: ReasonCode,
        reason: DetachReason,
        events: &mut Vec<AuthorityEvent>,
    ) {
        let why = self.reason(code, id, "");
        self.end_leases_of(id, &why, events);
        let owned: Vec<ChannelId> = self
            .parts
            .values()
            .filter(|p| &p.owner == id)
            .map(|p| p.channel)
            .collect();
        self.remove_parts(id, &owned, events);
        self.contract.remove(id);
        self.bindings.remove(id);
        self.roster.remove(id);
        self.seen.remove(id);
        self.needed.retain(|(p, _), _| p != id);
        self.recompute_demands(events);
        events.push(AuthorityEvent::Detached {
            participant: id.clone(),
            reason,
        });
        self.settle_after_release(id, "holder detached", events);
        self.barrier_gone(id, self.now_ns);
    }

    /// The owner changed `device`'s entry:
    /// its participants that embody a robot keep attached and lose every
    /// lease they hold; the others are detached, to re-attach under the new
    /// entry. Both with `ACCESS_CHANGED`. Returns what it caused and who it
    /// detached.
    pub fn set_access(
        &mut self,
        device: &str,
        access: DeviceAccess,
        now_ns: i64,
    ) -> (Vec<AuthorityEvent>, Vec<ParticipantId>) {
        self.now_ns = now_ns;
        self.config.access.set(device, access);
        let mut events = Vec::new();
        let mut detached = Vec::new();
        let members: Vec<ParticipantId> = self
            .roster
            .values()
            .filter(|p| p.device == device && p.present)
            .map(|p| p.id.clone())
            .collect();
        for id in members {
            if self.parts.values().any(|p| p.owner == id) {
                let why = self.reason(ReasonCode::AccessChanged, &id, "");
                self.end_leases_of(&id, &why, &mut events);
                self.settle_after_release(&id, "its device's entry changed", &mut events);
            } else {
                self.detach(
                    &id,
                    ReasonCode::AccessChanged,
                    DetachReason::AccessChanged,
                    &mut events,
                );
                detached.push(id);
            }
        }
        if !events.is_empty() {
            self.revision += 1;
        }
        (events, detached)
    }

    /// The owner revoked `device`: its entry goes, and
    /// every participant of it is detached (`REVOKED`), robots it embodies
    /// too. Returns what it caused and who it detached.
    pub fn revoke(
        &mut self,
        device: &str,
        now_ns: i64,
    ) -> (Vec<AuthorityEvent>, Vec<ParticipantId>) {
        self.now_ns = now_ns;
        self.config.access.remove(device);
        let members: Vec<ParticipantId> = self
            .roster
            .values()
            .filter(|p| p.device == device && p.present)
            .map(|p| p.id.clone())
            .collect();
        let mut events = Vec::new();
        for id in &members {
            self.detach(id, ReasonCode::Revoked, DetachReason::Revoked, &mut events);
        }
        if !events.is_empty() {
            self.revision += 1;
        }
        (events, members)
    }

    /// A new process attached as `id` (a new generation): the old one
    /// ends. A robot's restart is the identity half of an embodiment swap: a
    /// new epoch voids every lease, and ACTIVE becomes ARMED; no other mode
    /// changes, so a restart never stands in for an arm or a clear. Anyone
    /// else's leases end as a lost holder's: only its parts stop.
    pub(super) fn replace_generation(
        &mut self,
        id: &ParticipantId,
        events: &mut Vec<AuthorityEvent>,
    ) {
        if self.parts.values().any(|p| &p.owner == id) {
            // Its new process may be another embodiment: its own leases end
            // and, in an armed session, it waits to be re-armed by name.
            // Every other robot goes on.
            let why = self.reason(ReasonCode::VoidedByEpoch, id, "the robot restarted");
            self.stop_robot(id, ReasonCode::VoidedByEpoch, &why, events);
            if matches!(self.mode, Mode::Armed | Mode::Active) {
                self.hold(id, why, events);
            }
            self.settle_after_release(id, "its robot restarted", events);
        } else {
            let why = self.reason(
                ReasonCode::HolderDetached,
                id,
                "a new process attached under its id",
            );
            self.release_holder(id, &self.held_by(id), &why, events);
            self.settle_after_release(id, "a new process replaced the holder", events);
        }
    }

    pub(super) fn require_attached(
        &self,
        id: &ParticipantId,
    ) -> Result<&ParticipantInfo, CtlError> {
        self.roster
            .get(id)
            .filter(|p| p.present)
            .ok_or_else(|| CtlError::new(CtlStatus::NotAttached, format!("{id} is not attached")))
    }

    pub(super) fn require_role(
        &self,
        id: &ParticipantId,
        role: Role,
    ) -> Result<&ParticipantInfo, CtlError> {
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
}
