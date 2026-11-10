//! The part table: declaring a robot's parts, host-assigned channel ids
//! and command groups.

use alloc::collections::{BTreeMap, BTreeSet};

use super::{Authority, AuthorityEvent};
use crate::prelude::*;
use crate::wire::part::PartSpec;
use crate::wire::{ChannelId, CtlError, CtlStatus, Mode, ParticipantId, ReasonCode, Role};

impl Authority {
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
    pub(super) fn declare_parts(
        &mut self,
        from: &ParticipantId,
        spec_version: u32,
        robot_json: &str,
        can_seal: &[String],
        events: &mut Vec<AuthorityEvent>,
    ) -> Result<(), CtlError> {
        self.require_role(from, Role::Embodiment)?;
        let declared = crate::spec::normalise(from, spec_version, robot_json)?;
        let sealed: BTreeSet<String> = declared
            .iter()
            .filter(|p| self.config.seal.seals(from, p.path.as_str()))
            .map(|p| p.path.to_string())
            .collect();
        if let Some(path) = sealed.iter().find(|p| !can_seal.contains(p)) {
            return Err(CtlError::new(
                CtlStatus::Unsupported,
                format!(
                    "{from}/{path}: this session seals the part, and this robot process can't                      seal it"
                ),
            ));
        }
        let fingerprint = crate::spec::normalise::fingerprint(&declared);
        self.scene_keeps_map(from, &fingerprint)?;
        if !self.config.places.is_empty() {
            let mut session: Vec<PartSpec> = self
                .parts
                .values()
                .filter(|p| &p.owner != from)
                .cloned()
                .collect();
            session.extend(declared.iter().cloned());
            crate::spec::view::parts(&session, &self.config.places)
                .map_err(|why| CtlError::new(CtlStatus::Incompatible, why))?;
        }
        let current: BTreeMap<&str, &PartSpec> = self
            .parts
            .values()
            .filter(|p| &p.owner == from)
            .map(|p| (p.path.as_str(), p))
            .collect();
        let armed = matches!(self.mode, Mode::Armed | Mode::Active);
        // A part under a lease never changes; anything else may, armed or
        // not: a robot joins or comes back into an armed session.
        let refuse_leased = |path: &str| {
            CtlError::new(
                CtlStatus::InvalidTransition,
                format!("{path}: a commandable part is not re-declared while it is leased"),
            )
        };
        let mut keep = BTreeSet::new();
        let mut fresh = Vec::new();
        let mut commands = false;
        for mut spec in declared {
            match current.get(spec.path.as_str()) {
                Some(old) => {
                    spec.channel = old.channel;
                    spec.applied_channel = old.applied_channel;
                    if spec == **old {
                        keep.insert(old.channel);
                        continue;
                    }
                    if self.leases.contains_key(&old.channel) {
                        return Err(refuse_leased(spec.path.as_str()));
                    }
                    commands |= old.command.is_some() || spec.command.is_some();
                    fresh.push(spec);
                }
                None => {
                    commands |= spec.command.is_some();
                    fresh.push(spec);
                }
            }
        }
        let removed: Vec<ChannelId> = current
            .values()
            .filter(|p| !keep.contains(&p.channel))
            .map(|p| {
                if self.leases.contains_key(&p.channel) && !fresh.iter().any(|f| f.path == p.path) {
                    Err(refuse_leased(p.path.as_str()))
                } else {
                    Ok(p.channel)
                }
            })
            .collect::<Result<_, _>>()?;
        // New commands in an armed session wait for an arm, unless robots
        // re-arm on their own.
        if armed && commands && !self.config.rearm_robots {
            let why = self.reason(
                ReasonCode::Unspecified,
                from,
                "declared commandable parts while armed",
            );
            self.held.insert(from.clone(), why);
        }
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
        if let Some(info) = self.roster.get_mut(from) {
            info.left = None;
        }
        self.rejoined(from);
        for spec in self
            .parts
            .values()
            .filter(|p| &p.owner == from)
            .chain(&added)
        {
            let streams = [Some(spec.channel), spec.applied_channel];
            for channel in streams.into_iter().flatten() {
                if sealed.contains(spec.path.as_str()) {
                    self.sealed.insert(channel);
                } else {
                    self.sealed.remove(&channel);
                }
            }
        }
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
    pub(super) fn remove_parts(
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
                    self.sealed.remove(&stream);
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

    pub(super) fn commandable(&self, channel: ChannelId) -> Option<&PartSpec> {
        self.parts.get(&channel).filter(|p| p.command.is_some())
    }

    /// The `(robot, group)` of a commandable part.
    pub(super) fn group_of(&self, channel: ChannelId) -> Option<(&ParticipantId, &str)> {
        let part = self.parts.get(&channel)?;
        let group = part.command.as_ref()?.group.as_deref()?;
        Some((&part.owner, group))
    }

    /// The other members of `channel`'s command group.
    pub(super) fn group_peers(&self, channel: ChannelId) -> Vec<ChannelId> {
        let Some(group) = self.group_of(channel) else {
            return Vec::new();
        };
        self.parts
            .keys()
            .copied()
            .filter(|c| *c != channel && self.group_of(*c) == Some(group))
            .collect()
    }
}
