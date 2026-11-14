//! The host's key schedule, sans-IO: every key it drew,
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
//! - **Stream keys** (`k_stream`, one slot per part) go to the members
//!   whose device may read the part ([`Access::may_read`]), and rotate only
//!   when the slot's sender changes (a robot's new generation) or a device
//!   that holds one loses read permission ([`KeyIssuer::reauthorize`]):
//!   install the new
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
//! - **E-stop keys.** Each generation gets a `k_estop` when it takes
//!   effect: it sends under it (the key comes back in its `Attach` reply,
//!   [`KeyIssuer::estop_key`]) and every robot member receives it, so its
//!   edges latch the member's signal with no host. Nothing waits
//!   on their acknowledgements: until an edge has the key, it drops the
//!   signal and the ctl `Estop` still stops the session.
//!
//! Every key is 32 random bytes from the caller's [`Rng`], under an id from
//! one counter per host instance (shared with the handshake's keys).

use alloc::collections::{BTreeMap, BTreeSet};

use crate::authority::Access;
use crate::crypto::{Rng, RngError};
use crate::prelude::*;
use crate::wire::{
    ChannelId, ChannelPriority, CtlError, CtlStatus, IssuedKey, KeyBytes, KeyPurpose, KeysAck,
    LeaseToken, ParticipantId, PriorityClass, Security,
};

mod delivery;
mod grants;
mod streams;

/// How long a rotation waits for receivers to install a new version after
/// the host did, before it switches the sender.
pub const INSTALL_WAIT_NS: i64 = 500_000_000;
/// How long a grant waits for its robots to install its keys.
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

/// A participant generation taking effect ([`KeyIssuer::attach`]).
#[derive(Clone, Copy, Debug)]
pub struct Joining<'a> {
    /// Who.
    pub participant: &'a ParticipantId,
    /// Its generation.
    pub generation: u32,
    /// Its device.
    pub device: &'a str,
    /// It embodies a robot (its edges latch every member's e-stop).
    pub robot: bool,
}

/// One declared part's stream slot ([`KeyIssuer::sync_streams`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamSlot {
    /// The robot that sends it.
    pub robot: ParticipantId,
    /// The part's path.
    pub path: String,
    /// The part's `channel` and `applied_channel`.
    pub channels: Vec<ChannelId>,
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
    /// The robots whose every part the holder's device may read: only they
    /// get a `k_obs` for it (lockstep replies).
    pub reads_all: Vec<ParticipantId>,
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
    device: String,
    /// It embodies a robot: its edges receive every member's `k_estop`.
    robot: bool,
    /// Its generation's `k_estop`.
    estop: u32,
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
    /// The part's path (its stream's read permission).
    path: String,
    security: Security,
    /// The devices some version went to: a narrowing that takes one's read
    /// permission rotates the slot.
    readers: BTreeSet<String>,
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

    /// A participant generation took effect ([`Joining`]): it gets every stream key its device
    /// may read and a `k_estop` of its own, a robot also every member's
    /// `k_estop`, and a robot's streams rotate to it. Returns the pending
    /// grants an older generation's end failed.
    ///
    /// # Errors
    /// [`CtlError`] when no key could be drawn.
    pub fn attach(
        &mut self,
        rng: &mut dyn Rng,
        joining: Joining<'_>,
        access: &Access,
        now_ns: i64,
    ) -> Result<Vec<GrantId>, CtlError> {
        let Joining {
            participant,
            generation,
            device,
            robot,
        } = joining;
        let failed = match self.members.get(participant) {
            Some(m) if m.generation == generation => return Ok(Vec::new()),
            Some(_) => self.detach(participant),
            None => Vec::new(),
        };
        let mut estop = self.draw(rng, KeyPurpose::Estop, Security::Mac)?;
        estop.sender = Some(participant.clone());
        estop.generation = generation;
        let estop_id = estop.key_id;
        self.keys.insert(estop_id, estop);
        self.members.insert(
            participant.clone(),
            Member {
                generation,
                device: device.to_string(),
                robot,
                estop: estop_id,
            },
        );
        let others: Vec<(ParticipantId, bool, u32)> = self
            .members
            .iter()
            .filter(|(id, _)| *id != participant)
            .map(|(id, m)| (id.clone(), m.robot, m.estop))
            .collect();
        for (other, other_robot, other_estop) in others {
            if other_robot {
                self.queue(&other, estop_id, false);
            }
            if robot {
                self.queue(participant, other_estop, false);
            }
        }
        let slots: Vec<Slot> = self.slots.keys().cloned().collect();
        for slot in slots {
            if &slot.robot == participant {
                self.rotate(rng, &slot, generation, access, now_ns)?;
            } else if let Some(state) = self.slots.get_mut(&slot)
                && access.may_read(device, &slot.robot, &state.path)
            {
                state.readers.insert(device.to_string());
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
        if let Some(member) = self.members.remove(participant) {
            self.forget(&[member.estop]);
        }
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
}

#[cfg(test)]
mod tests;
