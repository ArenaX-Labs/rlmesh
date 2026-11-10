//! Controller bindings and the delivery contracts that come with them:
//! admission, the per-stream demand and its acknowledgement.

use alloc::collections::{BTreeMap, BTreeSet};

use super::{Authority, AuthorityEvent};
use crate::prelude::*;
use crate::spec::contract;
use crate::wire::part::PartSpec;
use crate::wire::{
    BindingRecord, ChannelId, CtlError, CtlStatus, Demand, Lease, ParticipantId, ReasonCode, Role,
};

/// The highest replan rate a binding declares.
const MAX_REPLAN_HZ: f64 = 10_000.0;

/// The longest action period a binding declares.
const MAX_ACTION_DT: core::time::Duration = core::time::Duration::from_secs(60);

impl Authority {
    pub(super) fn declare_binding(
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
    pub(super) fn stream_part(&self, channel: ChannelId) -> Option<(&PartSpec, bool)> {
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

    /// Admission: every contract names a stream once and is one the
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
    pub(super) fn recompute_demands(&mut self, events: &mut Vec<AuthorityEvent>) {
        let mut wanted: BTreeMap<ChannelId, Vec<&crate::wire::StreamContract>> = BTreeMap::new();
        for (participant, binding) in &self.bindings {
            if !self.is_attached(participant) {
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
            if !self.is_attached(participant) {
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

    pub(super) fn demand_ack(
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
}
