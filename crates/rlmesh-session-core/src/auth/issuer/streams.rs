//! Stream keys: one slot per part, its versions, and the rotation that
//! installs, sends and retires them.

use super::{
    INSTALL_WAIT_NS, KeyIssuer, PendingStep, Rotation, Slot, SlotState, Stage, StreamSlot,
};
use crate::authority::Access;
use crate::crypto::Rng;
use crate::prelude::*;
use crate::wire::{CtlError, KeyPurpose, ParticipantId, Security};
use alloc::collections::{BTreeMap, BTreeSet};

impl KeyIssuer {
    /// Keeps one stream slot per declared part (its `channel` and
    /// `applied_channel`), at each part's level: a new part's slot starts
    /// at version 1, a removed part's keys are retired.
    ///
    /// # Errors
    /// [`CtlError`] when no key could be drawn.
    pub fn sync_streams(
        &mut self,
        rng: &mut dyn Rng,
        streams: &[StreamSlot],
        access: &Access,
        now_ns: i64,
    ) -> Result<(), CtlError> {
        let wanted: BTreeMap<Slot, (&str, Security)> = streams
            .iter()
            .map(|stream| {
                (
                    Slot {
                        robot: stream.robot.clone(),
                        channels: stream.channels.clone(),
                    },
                    (stream.path.as_str(), stream.security),
                )
            })
            .collect();
        let gone: Vec<Slot> = self
            .slots
            .keys()
            .filter(|s| !wanted.contains_key(*s))
            .cloned()
            .collect();
        for slot in gone {
            let Some(state) = self.slots.remove(&slot) else {
                continue;
            };
            if let Some((_, newest)) = state.versions.last_key_value() {
                let members: Vec<ParticipantId> = self.members.keys().cloned().collect();
                for member in &members {
                    self.queue_retire(member, *newest);
                }
                self.host.retire.insert(*newest);
                self.host.sent_ns = None;
            }
            let ids: Vec<u32> = state.versions.values().copied().collect();
            self.forget(&ids);
        }
        for (slot, (path, security)) in wanted {
            if self.slots.contains_key(&slot) {
                continue;
            }
            let Some(generation) = self.members.get(&slot.robot).map(|m| m.generation) else {
                continue;
            };
            self.slots.insert(
                slot.clone(),
                SlotState {
                    path: path.to_string(),
                    security,
                    readers: BTreeSet::new(),
                    versions: BTreeMap::new(),
                    sender_generation: 0,
                    sending: None,
                    rotation: None,
                },
            );
            self.rotate(rng, &slot, generation, access, now_ns)?;
        }
        self.advance(now_ns);
        Ok(())
    }

    /// Prepares `slot`'s next version for sender generation `generation`,
    /// unless the slot already has one for it, and starts installing it at
    /// the members `access` lets read it.
    pub(super) fn rotate(
        &mut self,
        rng: &mut dyn Rng,
        slot: &Slot,
        generation: u32,
        access: &Access,
        now_ns: i64,
    ) -> Result<(), CtlError> {
        if self
            .slots
            .get(slot)
            .is_none_or(|s| s.sender_generation == generation)
        {
            return Ok(());
        }
        self.next_version(rng, slot, generation, access, now_ns)
    }

    /// Draws `slot`'s next version for sender generation `generation` and
    /// starts installing it at the members `access` lets read it. A
    /// rotation still under way is taken over: the new one retires every
    /// older version.
    fn next_version(
        &mut self,
        rng: &mut dyn Rng,
        slot: &Slot,
        generation: u32,
        access: &Access,
        now_ns: i64,
    ) -> Result<(), CtlError> {
        let Some(state) = self.slots.get(slot) else {
            return Ok(());
        };
        let path = state.path.clone();
        let security = state.security;
        let version = state.versions.keys().last().map_or(1, |v| v + 1);
        let retire = state.versions.values().last().copied();
        let mut key = self.draw(rng, KeyPurpose::Stream, security)?;
        key.sender = Some(slot.robot.clone());
        key.robot = Some(slot.robot.clone());
        key.channels = slot.channels.clone();
        key.version = version;
        key.generation = generation;
        let key_id = key.key_id;
        self.keys.insert(key_id, key);
        let receivers: BTreeSet<ParticipantId> = self
            .members
            .iter()
            .filter(|(m, member)| {
                *m != &slot.robot && access.may_read(&member.device, &slot.robot, &path)
            })
            .map(|(m, _)| m.clone())
            .collect();
        let readers: BTreeSet<String> = receivers
            .iter()
            .filter_map(|r| self.members.get(r).map(|m| m.device.clone()))
            .collect();
        for receiver in &receivers {
            self.queue(receiver, key_id, false);
        }
        self.host.install.insert(key_id, false);
        self.host.sent_ns = None;
        if let Some(state) = self.slots.get_mut(slot) {
            state.versions.insert(version, key_id);
            state.sender_generation = generation;
            state.readers = readers;
            state.rotation = Some(Rotation {
                version,
                key_id,
                stage: Stage::Install,
                started_ns: now_ns,
                receivers,
                retire,
            });
        }
        Ok(())
    }

    /// A device's entry changed: every slot a device that holds one of its
    /// versions may no longer read rotates, once, to the members still
    /// allowed. A slot is never rotated for its own sender.
    ///
    /// # Errors
    /// [`CtlError`] when no key could be drawn.
    pub fn reauthorize(
        &mut self,
        rng: &mut dyn Rng,
        access: &Access,
        now_ns: i64,
    ) -> Result<(), CtlError> {
        let narrowed: Vec<(Slot, u32)> = self
            .slots
            .iter()
            .filter(|(slot, state)| {
                state
                    .readers
                    .iter()
                    .any(|device| !access.may_read(device, &slot.robot, &state.path))
            })
            .map(|(slot, state)| (slot.clone(), state.sender_generation))
            .collect();
        for (slot, generation) in narrowed {
            self.next_version(rng, &slot, generation, access, now_ns)?;
        }
        self.advance(now_ns);
        Ok(())
    }

    /// Moves every rotation as far as its acknowledgements allow.
    pub(super) fn advance(&mut self, now_ns: i64) {
        let slots: Vec<Slot> = self
            .slots
            .iter()
            .filter(|(_, s)| s.rotation.is_some())
            .map(|(slot, _)| slot.clone())
            .collect();
        for slot in slots {
            while let Some(rotation) = self.slots.get(&slot).and_then(|s| s.rotation.clone()) {
                let waiting = self.waiting(&slot, &rotation);
                let next = match rotation.stage {
                    Stage::Install => {
                        let host_installed = !self.host.installing(rotation.key_id);
                        let timed_out =
                            now_ns.saturating_sub(rotation.started_ns) >= INSTALL_WAIT_NS;
                        (host_installed && (waiting.is_empty() || timed_out)).then_some(Stage::Send)
                    }
                    Stage::Send => waiting.is_empty().then_some(Stage::Retire),
                    Stage::Retire => waiting.is_empty().then_some(Stage::Retire),
                };
                let Some(next) = next else {
                    break;
                };
                match (rotation.stage, next) {
                    (Stage::Install, _) => {
                        self.queue(&slot.robot, rotation.key_id, true);
                        if let Some(state) = self.slots.get_mut(&slot) {
                            state.sending = Some(rotation.key_id);
                            if let Some(r) = &mut state.rotation {
                                r.stage = Stage::Send;
                                r.started_ns = now_ns;
                            }
                        }
                    }
                    (Stage::Send, _) => match rotation.retire {
                        Some(old) => {
                            let receivers: BTreeSet<ParticipantId> = self
                                .members
                                .keys()
                                .filter(|m| *m != &slot.robot)
                                .cloned()
                                .collect();
                            for receiver in &receivers {
                                self.queue_retire(receiver, old);
                            }
                            self.host.retire.insert(old);
                            self.host.sent_ns = None;
                            if let Some(r) =
                                self.slots.get_mut(&slot).and_then(|s| s.rotation.as_mut())
                            {
                                r.stage = Stage::Retire;
                                r.started_ns = now_ns;
                                r.receivers = receivers;
                            }
                        }
                        None => self.finish(&slot, rotation.version),
                    },
                    (Stage::Retire, _) => {
                        if self.host.retire.contains(&rotation.retire.unwrap_or(0)) {
                            break;
                        }
                        self.finish(&slot, rotation.version);
                    }
                }
            }
        }
    }

    /// A rotation completed: the versions below it are gone everywhere.
    fn finish(&mut self, slot: &Slot, version: u32) {
        let Some(state) = self.slots.get_mut(slot) else {
            return;
        };
        state.rotation = None;
        let older: Vec<u32> = state
            .versions
            .iter()
            .filter(|(v, _)| **v < version)
            .map(|(_, id)| *id)
            .collect();
        state.versions.retain(|v, _| *v >= version);
        self.forget(&older);
    }

    /// Who a rotation's current stage waits for.
    fn waiting(&self, slot: &Slot, rotation: &Rotation) -> Vec<ParticipantId> {
        let pending = |who: &ParticipantId| -> bool {
            self.members.contains_key(who)
                && self.outbox.get(who).is_some_and(|o| match rotation.stage {
                    Stage::Install => o.installing(rotation.key_id),
                    Stage::Send => o.installing(rotation.key_id),
                    Stage::Retire => rotation.retire.is_some_and(|id| o.retire.contains(&id)),
                })
        };
        match rotation.stage {
            Stage::Send => pending(&slot.robot)
                .then(|| slot.robot.clone())
                .into_iter()
                .collect(),
            Stage::Install | Stage::Retire => rotation
                .receivers
                .iter()
                .filter(|r| pending(r))
                .cloned()
                .collect(),
        }
    }

    /// Every rotation step still waiting, and for whom: what a revocation
    /// report names.
    pub fn pending(&self) -> Vec<PendingStep> {
        self.slots
            .iter()
            .filter_map(|(slot, state)| {
                let rotation = state.rotation.as_ref()?;
                Some(PendingStep {
                    robot: slot.robot.clone(),
                    channels: slot.channels.clone(),
                    version: rotation.version,
                    stage: rotation.stage.name(),
                    waiting: self.waiting(slot, rotation),
                })
            })
            .collect()
    }
}
