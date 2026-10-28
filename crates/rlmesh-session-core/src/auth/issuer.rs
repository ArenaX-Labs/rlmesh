//! The host's key schedule (design §8.4, §8.5), sans-IO: every key it drew,
//! who holds it, what each member still has to acknowledge, stream key
//! rotations and the pending grants whose keys are on their way to the
//! robots.
//!
//! - **Deliveries.** Each member has an outbox of keys to install (or to
//!   send under) and keys to retire. [`KeyIssuer::deliveries`] hands out a
//!   `KeyDelivery` per member with something new, and again every
//!   [`RETRY_NS`] until its `KeysAck` clears it. A missed acknowledgement
//!   never detaches anyone and never counts as one: the step waits, and
//!   [`KeyIssuer::pending`] says who it waits for. The host's own receivers
//!   are one more outbox ([`KeyIssuer::host_delivery`]).
//! - **Stream keys** (`k_stream`, one slot per part) rotate only when the
//!   slot's sender changes (a robot's new generation): install the new
//!   version at every receiver (the host first, then at most
//!   [`INSTALL_WAIT_NS`] for the rest), send it to the robot, then retire
//!   the older versions everywhere. A new part gets version 1 the same way,
//!   with nothing to retire; a removed part's keys are retired.
//! - **Grants.** A grant's `k_cmd` (per robot and transport priority) and
//!   `k_obs` (per robot) go to each robot first; once every robot
//!   acknowledged, the host re-checks the grant and commits it, handing the
//!   holder its keys in the reply ([`KeyIssuer::commit_grant`]), or refuses
//!   it and retires them ([`KeyIssuer::refuse_grant`]). One device has at
//!   most [`DEVICE_PENDING`] grants pending and one robot
//!   [`ROBOT_PENDING`]; a full robot evicts the newest pending grant of the
//!   lowest class below the new one (`PREEMPTED`).
//!
//! Every key is 32 random bytes from the caller's [`Rng`], under an id from
//! one counter per host instance (shared with the handshake's keys).

use alloc::collections::{BTreeMap, BTreeSet};

use crate::crypto::{Rng, RngError};
use crate::prelude::*;
use crate::wire::{
    ChannelId, ChannelPriority, CtlError, CtlStatus, IssuedKey, KeyBytes, KeyDelivery, KeyPurpose,
    KeysAck, LeaseToken, ParticipantId, PriorityClass, Security,
};

/// How long a rotation waits for receivers to install a new version after
/// the host did, before it switches the sender (§8.5 step 2).
pub const INSTALL_WAIT_NS: i64 = 500_000_000;
/// How long a grant waits for its robots to install its keys (§8.5).
pub const GRANT_WAIT_NS: i64 = 2_000_000_000;
/// How often an unacknowledged delivery is sent again.
pub const RETRY_NS: i64 = 250_000_000;
/// Pending grants per device.
pub const DEVICE_PENDING: usize = 2;
/// Pending grants per robot.
pub const ROBOT_PENDING: usize = 4;

/// A pending grant.
pub type GrantId = u64;

/// One part of a grant: a lease on `channel` of `robot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrantPart {
    /// The robot that owns the part.
    pub robot: ParticipantId,
    /// The part.
    pub channel: ChannelId,
    /// Its transport priority (one `k_cmd` per priority).
    pub priority: ChannelPriority,
    /// Its level.
    pub security: Security,
}

/// A grant to prepare: the parts its new token covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GrantRequest {
    /// The requester.
    pub holder: ParticipantId,
    /// Its generation.
    pub holder_generation: u32,
    /// Its device.
    pub device: String,
    /// Its class.
    pub class: PriorityClass,
    /// The token the grant mints.
    pub token: LeaseToken,
    /// The parts under that token.
    pub parts: Vec<GrantPart>,
}

/// A prepared grant: its id, and the pending grants it evicted (each to be
/// refused `PREEMPTED`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    /// The grant.
    pub id: GrantId,
    /// The grants it evicted.
    pub evicted: Vec<GrantId>,
}

/// A committed grant: the holder's keys and each part's `k_cmd` id.
#[derive(Debug)]
pub struct Committed {
    /// For the holder's reply.
    pub keys: Vec<IssuedKey>,
    /// The `Lease.key_id` of each part.
    pub key_ids: Vec<(ChannelId, u32)>,
}

/// A step still waiting for acknowledgements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingStep {
    /// The robot whose stream rotates.
    pub robot: ParticipantId,
    /// The stream's parts.
    pub channels: Vec<ChannelId>,
    /// The version it rotates to.
    pub version: u32,
    /// `install`, `send` or `retire`.
    pub stage: &'static str,
    /// Who hasn't acknowledged.
    pub waiting: Vec<ParticipantId>,
}

#[derive(Clone, Debug)]
struct Member {
    generation: u32,
}

#[derive(Clone, Debug, Default)]
struct Outbox {
    /// Key id → deliver it with `send`.
    install: BTreeMap<u32, bool>,
    retire: BTreeSet<u32>,
    /// When it was last sent; `None`: something new to send.
    sent_ns: Option<i64>,
}

impl Outbox {
    fn is_empty(&self) -> bool {
        self.install.is_empty() && self.retire.is_empty()
    }

    fn take_ack(&mut self, ack: &KeysAck) {
        self.install.retain(|id, send| {
            !if *send {
                ack.sending.contains(id)
            } else {
                ack.installed.contains(id)
            }
        });
        self.retire.retain(|id| !ack.retired.contains(id));
    }

    fn installing(&self, id: u32) -> bool {
        self.install.contains_key(&id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Install,
    Send,
    Retire,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Install => "install",
            Self::Send => "send",
            Self::Retire => "retire",
        }
    }
}

#[derive(Clone, Debug)]
struct Rotation {
    version: u32,
    key_id: u32,
    stage: Stage,
    started_ns: i64,
    /// The receivers frozen when the stage began.
    receivers: BTreeSet<ParticipantId>,
    /// The newest older version's key, retired at the end.
    retire: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Slot {
    robot: ParticipantId,
    channels: Vec<ChannelId>,
}

#[derive(Clone, Debug)]
struct SlotState {
    security: Security,
    /// Version → key id, oldest first.
    versions: BTreeMap<u32, u32>,
    /// The robot generation the newest version is for.
    sender_generation: u32,
    /// The key the robot was told to send under.
    sending: Option<u32>,
    rotation: Option<Rotation>,
}

/// Who a lease key belongs to.
#[derive(Clone, Debug)]
struct LeaseKey {
    holder: ParticipantId,
    holder_generation: u32,
    robot: ParticipantId,
    robot_generation: u32,
    committed: bool,
}

#[derive(Clone, Debug)]
struct Pending {
    request: GrantRequest,
    device: String,
    keys: Vec<u32>,
    robots: BTreeSet<ParticipantId>,
    deadline_ns: i64,
}

/// The host's key schedule: see the module docs.
#[derive(Debug, Default)]
pub struct KeyIssuer {
    last_key_id: u32,
    members: BTreeMap<ParticipantId, Member>,
    keys: BTreeMap<u32, IssuedKey>,
    leases: BTreeMap<u32, LeaseKey>,
    slots: BTreeMap<Slot, SlotState>,
    outbox: BTreeMap<ParticipantId, Outbox>,
    host: Outbox,
    grants: BTreeMap<GrantId, Pending>,
    last_grant: GrantId,
}

fn exhausted() -> CtlError {
    CtlError::new(
        CtlStatus::Unsupported,
        "this host instance has issued every key id; restart it",
    )
}

impl KeyIssuer {
    /// The next key id, from the one counter of this host instance.
    pub fn next_key_id(&mut self) -> Option<u32> {
        self.last_key_id = self.last_key_id.checked_add(1)?;
        Some(self.last_key_id)
    }

    fn draw(
        &mut self,
        rng: &mut dyn Rng,
        purpose: KeyPurpose,
        security: Security,
    ) -> Result<IssuedKey, CtlError> {
        let key_id = self.next_key_id().ok_or_else(exhausted)?;
        let mut bytes = KeyBytes([0; 32]);
        rng.fill_bytes(&mut bytes.0).map_err(|RngError| {
            CtlError::new(CtlStatus::Unsupported, "the random number generator failed")
        })?;
        Ok(IssuedKey {
            key_id,
            purpose,
            sender: None,
            robot: None,
            channels: Vec::new(),
            token: LeaseToken(0),
            version: 0,
            key: bytes,
            send: false,
            security,
            generation: 0,
        })
    }

    fn queue(&mut self, to: &ParticipantId, key_id: u32, send: bool) {
        if self.members.contains_key(to) {
            let outbox = self.outbox.entry(to.clone()).or_default();
            outbox.install.insert(key_id, send);
            outbox.sent_ns = None;
        }
    }

    fn queue_retire(&mut self, to: &ParticipantId, key_id: u32) {
        if self.members.contains_key(to) {
            let outbox = self.outbox.entry(to.clone()).or_default();
            outbox.retire.insert(key_id);
            outbox.sent_ns = None;
        }
    }

    /// Forgets keys, and every delivery of them not yet acknowledged.
    fn forget(&mut self, ids: &[u32]) {
        for id in ids {
            self.keys.remove(id);
            self.leases.remove(id);
        }
        for outbox in self.outbox.values_mut().chain([&mut self.host]) {
            outbox.install.retain(|id, _| !ids.contains(id));
        }
    }

    /// Generation `generation` of `participant` took effect: it gets every
    /// stream key it receives, and a robot's streams rotate to it. Returns
    /// the pending grants an older generation's end failed.
    ///
    /// # Errors
    /// [`CtlError`] when no key could be drawn.
    pub fn attach(
        &mut self,
        rng: &mut dyn Rng,
        participant: &ParticipantId,
        generation: u32,
        now_ns: i64,
    ) -> Result<Vec<GrantId>, CtlError> {
        let failed = match self.members.get(participant) {
            Some(m) if m.generation == generation => return Ok(Vec::new()),
            Some(_) => self.detach(participant),
            None => Vec::new(),
        };
        self.members
            .insert(participant.clone(), Member { generation });
        let slots: Vec<Slot> = self.slots.keys().cloned().collect();
        for slot in slots {
            if &slot.robot == participant {
                self.rotate(rng, &slot, generation, now_ns)?;
            } else if let Some(state) = self.slots.get(&slot) {
                let versions: Vec<u32> = state.versions.values().copied().collect();
                for id in versions {
                    self.queue(participant, id, false);
                }
            }
        }
        Ok(failed)
    }

    /// `participant`'s generation ended (detached, superseded or lost): it
    /// leaves every pending set at once, as if it had acknowledged. Returns
    /// the pending grants that needed it (as holder or robot), now failed.
    pub fn detach(&mut self, participant: &ParticipantId) -> Vec<GrantId> {
        self.members.remove(participant);
        self.outbox.remove(participant);
        let failed: Vec<GrantId> = self
            .grants
            .iter()
            .filter(|(_, g)| &g.request.holder == participant || g.robots.contains(participant))
            .map(|(id, _)| *id)
            .collect();
        for id in &failed {
            self.refuse_grant(*id);
        }
        failed
    }

    /// Keeps one stream slot per declared part (its `channel` and
    /// `applied_channel`), at each part's level: a new part's slot starts
    /// at version 1, a removed part's keys are retired.
    ///
    /// # Errors
    /// [`CtlError`] when no key could be drawn.
    pub fn sync_streams(
        &mut self,
        rng: &mut dyn Rng,
        streams: &[(ParticipantId, Vec<ChannelId>, Security)],
        now_ns: i64,
    ) -> Result<(), CtlError> {
        let wanted: BTreeMap<Slot, Security> = streams
            .iter()
            .map(|(robot, channels, security)| {
                (
                    Slot {
                        robot: robot.clone(),
                        channels: channels.clone(),
                    },
                    *security,
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
        for (slot, security) in wanted {
            if self.slots.contains_key(&slot) {
                continue;
            }
            let Some(generation) = self.members.get(&slot.robot).map(|m| m.generation) else {
                continue;
            };
            self.slots.insert(
                slot.clone(),
                SlotState {
                    security,
                    versions: BTreeMap::new(),
                    sender_generation: 0,
                    sending: None,
                    rotation: None,
                },
            );
            self.rotate(rng, &slot, generation, now_ns)?;
        }
        self.advance(now_ns);
        Ok(())
    }

    /// Prepares `slot`'s next version for sender generation `generation`
    /// and starts installing it. A rotation still under way is taken over:
    /// the new one retires every older version.
    fn rotate(
        &mut self,
        rng: &mut dyn Rng,
        slot: &Slot,
        generation: u32,
        now_ns: i64,
    ) -> Result<(), CtlError> {
        let Some(state) = self.slots.get(slot) else {
            return Ok(());
        };
        if state.sender_generation == generation {
            return Ok(());
        }
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
        // TODO: every member reads every stream until per-device read
        // permission lands (step 8): narrow the receivers to it then, and
        // rotate a slot when a device loses it.
        let receivers: BTreeSet<ParticipantId> = self
            .members
            .keys()
            .filter(|m| *m != &slot.robot)
            .cloned()
            .collect();
        for receiver in &receivers {
            self.queue(receiver, key_id, false);
        }
        self.host.install.insert(key_id, false);
        self.host.sent_ns = None;
        if let Some(state) = self.slots.get_mut(slot) {
            state.versions.insert(version, key_id);
            state.sender_generation = generation;
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

    /// Moves every rotation as far as its acknowledgements allow.
    fn advance(&mut self, now_ns: i64) {
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
    /// report names (§8.5).
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

    /// What generation `generation` of `participant` acknowledged.
    pub fn ack(
        &mut self,
        participant: &ParticipantId,
        generation: u32,
        ack: &KeysAck,
        now_ns: i64,
    ) {
        if self
            .members
            .get(participant)
            .is_none_or(|m| m.generation != generation)
        {
            return;
        }
        if let Some(outbox) = self.outbox.get_mut(participant) {
            outbox.take_ack(ack);
        }
        self.advance(now_ns);
    }

    /// The host's own receivers' delivery, if anything changed: apply it
    /// to the host's keyring and [`Self::host_ack`] the result.
    pub fn host_delivery(&mut self) -> Option<KeyDelivery> {
        if self.host.is_empty() || self.host.sent_ns.is_some() {
            return None;
        }
        self.host.sent_ns = Some(0);
        Some(self.delivery(&self.host))
    }

    /// What the host's own keyring did with its delivery.
    pub fn host_ack(&mut self, ack: &KeysAck, now_ns: i64) {
        self.host.take_ack(ack);
        self.host.sent_ns = None;
        self.advance(now_ns);
    }

    fn delivery(&self, outbox: &Outbox) -> KeyDelivery {
        KeyDelivery {
            keys: outbox
                .install
                .iter()
                .filter_map(|(id, send)| {
                    self.keys.get(id).map(|key| IssuedKey {
                        send: *send,
                        ..key.clone()
                    })
                })
                .collect(),
            retire_through: outbox.retire.iter().copied().collect(),
        }
    }

    /// The deliveries to send now: each member's outbox with something new,
    /// or unacknowledged for [`RETRY_NS`]. Also moves rotations whose
    /// install wait ran out.
    pub fn deliveries(&mut self, now_ns: i64) -> Vec<(ParticipantId, KeyDelivery)> {
        self.advance(now_ns);
        let due: Vec<ParticipantId> = self
            .outbox
            .iter()
            .filter(|(_, o)| {
                !o.is_empty()
                    && o.sent_ns
                        .is_none_or(|sent| now_ns.saturating_sub(sent) >= RETRY_NS)
            })
            .map(|(id, _)| id.clone())
            .collect();
        due.into_iter()
            .filter_map(|id| {
                let delivery = self.delivery(self.outbox.get(&id)?);
                if let Some(o) = self.outbox.get_mut(&id) {
                    o.sent_ns = Some(now_ns);
                }
                Some((id, delivery))
            })
            .collect()
    }

    /// Every key issued to generation `generation` of `participant` (with
    /// `send` where it is the sender), for its `GetKeys`.
    pub fn get_keys(&self, participant: &ParticipantId, generation: u32) -> Vec<IssuedKey> {
        if self
            .members
            .get(participant)
            .is_none_or(|m| m.generation != generation)
        {
            return Vec::new();
        }
        let mut out = Vec::new();
        let with =
            |id: &u32, send: bool| self.keys.get(id).map(|k| IssuedKey { send, ..k.clone() });
        for (slot, state) in &self.slots {
            if &slot.robot == participant {
                if let Some(id) = state.sending
                    && state.sender_generation == generation
                {
                    out.extend(with(&id, true));
                }
            } else {
                out.extend(state.versions.values().filter_map(|id| with(id, false)));
            }
        }
        for (id, lease) in &self.leases {
            let Some(key) = self.keys.get(id) else {
                continue;
            };
            let sends = key.sender.as_ref() == Some(participant) && key.generation == generation;
            let holder = lease.committed
                && &lease.holder == participant
                && lease.holder_generation == generation;
            let robot = &lease.robot == participant && lease.robot_generation == generation;
            if holder || robot {
                out.extend(with(id, sends));
            }
        }
        out
    }

    /// Prepares a grant: draws its keys and sends them to its robots.
    ///
    /// # Errors
    /// `LEASE_BUSY` when the device has [`DEVICE_PENDING`] grants pending or
    /// a robot is full of grants of an equal or higher class; `NOT_ATTACHED`
    /// for a robot that isn't.
    pub fn prepare_grant(
        &mut self,
        rng: &mut dyn Rng,
        request: GrantRequest,
        now_ns: i64,
    ) -> Result<Prepared, CtlError> {
        let busy = |message: String| CtlError::new(CtlStatus::LeaseBusy, message);
        let device = request.device.clone();
        if self.grants.values().filter(|g| g.device == device).count() >= DEVICE_PENDING {
            return Err(busy(format!(
                "device {device} already has {DEVICE_PENDING} grants pending"
            )));
        }
        let robots: BTreeSet<ParticipantId> =
            request.parts.iter().map(|p| p.robot.clone()).collect();
        let mut generations = BTreeMap::new();
        for robot in &robots {
            let generation = self
                .members
                .get(robot)
                .map(|m| m.generation)
                .ok_or_else(|| {
                    CtlError::new(
                        CtlStatus::NotAttached,
                        format!("robot {robot} is not attached"),
                    )
                })?;
            generations.insert(robot.clone(), generation);
        }
        let mut evict = Vec::new();
        for robot in &robots {
            let pending: Vec<(&GrantId, &Pending)> = self
                .grants
                .iter()
                .filter(|(id, g)| g.robots.contains(robot) && !evict.contains(*id))
                .collect();
            if pending.len() < ROBOT_PENDING {
                continue;
            }
            // The newest pending grant of the lowest class below this one.
            let victim = pending
                .iter()
                .filter(|(_, g)| g.request.class < request.class)
                .min_by_key(|(id, g)| (g.request.class, core::cmp::Reverse(**id)))
                .map(|(id, _)| **id)
                .ok_or_else(|| {
                    busy(format!(
                        "robot {robot} already has {ROBOT_PENDING} grants pending"
                    ))
                })?;
            evict.push(victim);
        }
        for id in &evict {
            self.refuse_grant(*id);
        }
        let security = if request.parts.iter().any(|p| p.security == Security::Seal) {
            Security::Seal
        } else {
            Security::Sign
        };
        let mut keys = Vec::new();
        for robot in &robots {
            let robot_generation = generations[robot];
            let mut by_priority: BTreeMap<u8, Vec<ChannelId>> = BTreeMap::new();
            for part in request.parts.iter().filter(|p| &p.robot == robot) {
                by_priority
                    .entry(part.priority as u8)
                    .or_default()
                    .push(part.channel);
            }
            let lease = LeaseKey {
                holder: request.holder.clone(),
                holder_generation: request.holder_generation,
                robot: robot.clone(),
                robot_generation,
                committed: false,
            };
            for channels in by_priority.into_values() {
                let mut key = self.draw(rng, KeyPurpose::Command, security)?;
                key.sender = Some(request.holder.clone());
                key.generation = request.holder_generation;
                key.robot = Some(robot.clone());
                key.channels = channels;
                key.token = request.token;
                keys.push(key.key_id);
                self.queue(robot, key.key_id, false);
                self.leases.insert(key.key_id, lease.clone());
                self.keys.insert(key.key_id, key);
            }
            let mut obs = self.draw(rng, KeyPurpose::LockstepReply, security)?;
            obs.sender = Some(robot.clone());
            obs.generation = robot_generation;
            obs.robot = Some(robot.clone());
            obs.token = request.token;
            keys.push(obs.key_id);
            self.queue(robot, obs.key_id, true);
            self.leases.insert(obs.key_id, lease);
            self.keys.insert(obs.key_id, obs);
        }
        self.last_grant += 1;
        let id = self.last_grant;
        self.grants.insert(
            id,
            Pending {
                request,
                device,
                keys,
                robots,
                deadline_ns: now_ns.saturating_add(GRANT_WAIT_NS),
            },
        );
        Ok(Prepared { id, evicted: evict })
    }

    /// The pending grants that can be decided: `Ok` once every robot
    /// installed its keys, `Err` when that took over [`GRANT_WAIT_NS`].
    pub fn ready_grants(&self, now_ns: i64) -> Vec<(GrantId, Result<(), CtlError>)> {
        self.grants
            .iter()
            .filter_map(|(id, grant)| {
                let installed = grant.keys.iter().all(|key| {
                    self.leases.get(key).is_some_and(|lease| {
                        self.outbox
                            .get(&lease.robot)
                            .is_none_or(|o| !o.installing(*key))
                    })
                });
                if installed {
                    Some((*id, Ok(())))
                } else if now_ns >= grant.deadline_ns {
                    Some((
                        *id,
                        Err(CtlError::new(
                            CtlStatus::LeaseBusy,
                            "a robot did not install the lease's keys in time",
                        )),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// The prepared request of grant `id`.
    pub fn grant(&self, id: GrantId) -> Option<&GrantRequest> {
        self.grants.get(&id).map(|g| &g.request)
    }

    /// Commits grant `id`: the holder's keys (it sends under each `k_cmd`
    /// and receives the `k_obs`) and each part's `k_cmd` id.
    pub fn commit_grant(&mut self, id: GrantId) -> Option<Committed> {
        let grant = self.grants.remove(&id)?;
        let mut committed = Committed {
            keys: Vec::new(),
            key_ids: Vec::new(),
        };
        for key_id in &grant.keys {
            if let Some(lease) = self.leases.get_mut(key_id) {
                lease.committed = true;
            }
            let Some(key) = self.keys.get(key_id) else {
                continue;
            };
            let command = key.purpose == KeyPurpose::Command;
            if command {
                committed
                    .key_ids
                    .extend(key.channels.iter().map(|c| (*c, key.key_id)));
            }
            committed.keys.push(IssuedKey {
                send: command,
                ..key.clone()
            });
        }
        Some(committed)
    }

    /// Refuses grant `id`: its keys are retired at its robots.
    pub fn refuse_grant(&mut self, id: GrantId) {
        let Some(grant) = self.grants.remove(&id) else {
            return;
        };
        for key_id in &grant.keys {
            if let Some(robot) = self.leases.get(key_id).map(|l| l.robot.clone()) {
                self.queue_retire(&robot, *key_id);
            }
        }
        self.forget(&grant.keys);
    }

    /// Drops the committed lease keys whose token `live` says ended (their
    /// holders discard them from authority state).
    pub fn end_tokens(&mut self, live: impl Fn(LeaseToken) -> bool) {
        let ended: Vec<u32> = self
            .leases
            .iter()
            .filter(|(id, lease)| {
                lease.committed && self.keys.get(id).is_some_and(|k| !live(k.token))
            })
            .map(|(id, _)| *id)
            .collect();
        self.forget(&ended);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::keyring::Keyring;
    use crate::crypto::test_rng::TestRng;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    /// Delivers what is due and acknowledges it, except at `deaf`.
    fn exchange(
        issuer: &mut KeyIssuer,
        rings: &mut BTreeMap<ParticipantId, Keyring>,
        deaf: &[&str],
        now: i64,
    ) {
        if let Some(delivery) = issuer.host_delivery() {
            let ack = KeysAck {
                installed: delivery.keys.iter().map(|k| k.key_id).collect(),
                sending: Vec::new(),
                retired: delivery.retire_through.clone(),
            };
            issuer.host_ack(&ack, now);
        }
        for (who, delivery) in issuer.deliveries(now) {
            let Some(ring) = rings.get_mut(&who) else {
                continue;
            };
            let ack = ring.apply(&delivery, now);
            if !deaf.contains(&who.as_str()) {
                let generation = issuer.members[&who].generation;
                issuer.ack(&who, generation, &ack, now);
            }
        }
    }

    fn arm_stream() -> Vec<(ParticipantId, Vec<ChannelId>, Security)> {
        vec![(pid("arm"), vec![ChannelId(1), ChannelId(2)], Security::Sign)]
    }

    #[test]
    fn a_robots_new_generation_rotates_its_streams_and_retires_the_old_version() {
        let mut rng = TestRng::new(5);
        let mut issuer = KeyIssuer::default();
        let mut rings = BTreeMap::new();
        for (who, generation) in [("arm", 1), ("policy", 2)] {
            issuer
                .attach(&mut rng, &pid(who), generation, 0)
                .expect("attach");
            rings.insert(pid(who), Keyring::new(Some(pid(who)), generation));
        }
        issuer
            .sync_streams(&mut rng, &arm_stream(), 0)
            .expect("sync");
        for t in 0..4 {
            exchange(&mut issuer, &mut rings, &[], t);
        }
        assert!(issuer.pending().is_empty(), "version 1 is in place");
        assert!(rings[&pid("arm")].sends_stream(ChannelId(1)));
        let v1: Vec<u32> = rings[&pid("policy")]
            .scopes()
            .map(|(s, _)| s.key_id)
            .collect();

        // The robot restarts: generation 3 takes over its slot at version 2.
        issuer.attach(&mut rng, &pid("arm"), 3, 10).expect("attach");
        rings.insert(pid("arm"), Keyring::new(Some(pid("arm")), 3));
        exchange(&mut issuer, &mut rings, &[], 11);
        exchange(&mut issuer, &mut rings, &[], 12);
        exchange(&mut issuer, &mut rings, &[], 13);
        assert!(issuer.pending().is_empty());
        let arm = &rings[&pid("arm")];
        let sending: Vec<(u32, bool)> = arm.scopes().map(|(s, tx)| (s.version, tx)).collect();
        assert_eq!(sending, vec![(2, true)]);
        let policy: Vec<u32> = rings[&pid("policy")]
            .scopes()
            .map(|(s, _)| s.version)
            .collect();
        assert_eq!(policy, vec![2], "retire_through dropped version 1 ({v1:?})");
        // Recovery hands the new generation its sending key only.
        let recovered = issuer.get_keys(&pid("arm"), 3);
        assert_eq!(recovered.len(), 1);
        assert!(recovered[0].send && recovered[0].version == 2);
        assert!(
            issuer.get_keys(&pid("arm"), 1).is_empty(),
            "an ended generation"
        );
    }

    #[test]
    fn a_missed_ack_leaves_the_retire_pending_and_never_detaches() {
        let mut rng = TestRng::new(6);
        let mut issuer = KeyIssuer::default();
        let mut rings = BTreeMap::new();
        for (who, generation) in [("arm", 1), ("policy", 2), ("board", 3)] {
            issuer
                .attach(&mut rng, &pid(who), generation, 0)
                .expect("attach");
            rings.insert(pid(who), Keyring::new(Some(pid(who)), generation));
        }
        issuer
            .sync_streams(&mut rng, &arm_stream(), 0)
            .expect("sync");
        for t in 0..4 {
            exchange(&mut issuer, &mut rings, &[], t);
        }
        issuer.attach(&mut rng, &pid("arm"), 4, 10).expect("attach");
        rings.insert(pid("arm"), Keyring::new(Some(pid("arm")), 4));
        // The board never acknowledges: install waits 500 ms, then the
        // sender switches; the retire stays open, naming the board.
        exchange(&mut issuer, &mut rings, &["board"], 11);
        assert_eq!(issuer.pending()[0].stage, "install");
        assert_eq!(issuer.pending()[0].waiting, vec![pid("board")]);
        let later = 11 + INSTALL_WAIT_NS;
        for t in 0..3 {
            exchange(&mut issuer, &mut rings, &["board"], later + t * RETRY_NS);
        }
        let pending = issuer.pending();
        assert_eq!(pending.len(), 1);
        assert_eq!((pending[0].stage, pending[0].version), ("retire", 2));
        assert_eq!(pending[0].waiting, vec![pid("board")]);
        assert!(rings[&pid("arm")].sends_stream(ChannelId(1)));
        assert!(issuer.members.contains_key(&pid("board")), "still attached");
        // Once it acknowledges (a retry reaches it), the rotation completes.
        exchange(&mut issuer, &mut rings, &[], later + 10 * RETRY_NS);
        assert!(issuer.pending().is_empty());
    }

    #[test]
    fn grants_are_capped_per_device_and_robot_and_a_higher_class_evicts_the_newest_lowest() {
        let mut rng = TestRng::new(7);
        let mut issuer = KeyIssuer::default();
        issuer.attach(&mut rng, &pid("arm"), 1, 0).expect("attach");
        let request = |holder: &str, device: &str, class, token| GrantRequest {
            holder: pid(holder),
            holder_generation: 1,
            device: device.into(),
            class,
            token: LeaseToken(token),
            parts: vec![GrantPart {
                robot: pid("arm"),
                channel: ChannelId(1),
                priority: ChannelPriority::Realtime,
                security: Security::Sign,
            }],
        };
        let model = PriorityClass::Model;
        let a = issuer.prepare_grant(&mut rng, request("a", "d1", model, 1), 0);
        let b = issuer.prepare_grant(&mut rng, request("b", "d1", model, 2), 0);
        let busy = issuer.prepare_grant(&mut rng, request("c", "d1", model, 3), 0);
        assert_eq!(busy.expect_err("third").status, CtlStatus::LeaseBusy);
        let c = issuer.prepare_grant(&mut rng, request("c", "d2", model, 3), 0);
        let d = issuer.prepare_grant(&mut rng, request("d", "d2", model, 4), 0);
        let (a, b, c, d) = (a.expect("a"), b.expect("b"), c.expect("c"), d.expect("d"));
        let full = issuer.prepare_grant(&mut rng, request("e", "d3", model, 5), 0);
        assert_eq!(full.expect_err("robot full").status, CtlStatus::LeaseBusy);
        let operator =
            issuer.prepare_grant(&mut rng, request("o", "d3", PriorityClass::Operator, 6), 0);
        assert_eq!(
            operator.expect("evicts").evicted,
            vec![d.id],
            "the newest model grant"
        );
        assert!(issuer.grant(d.id).is_none());
        // Nothing installed: they all time out together.
        let decided = issuer.ready_grants(GRANT_WAIT_NS);
        assert_eq!(decided.len(), 4);
        assert!(decided.iter().all(|(_, r)| r.is_err()));
        let _ = (a, b, c);
    }
}
