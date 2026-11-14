//! Deliveries: each member's outbox handed out and acknowledged, the
//! host's own, e-stop keys and `GetKeys`.

use super::{KeyIssuer, Outbox, RETRY_NS};
use crate::authority::Access;
use crate::prelude::*;
use crate::wire::{IssuedKey, KeyDelivery, KeysAck, ParticipantId};

impl KeyIssuer {
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

    /// Generation `generation` of `participant`'s own `k_estop`, to send
    /// under: what its `Attach` reply carries.
    pub fn estop_key(&self, participant: &ParticipantId, generation: u32) -> Option<IssuedKey> {
        let member = self
            .members
            .get(participant)
            .filter(|m| m.generation == generation)?;
        self.keys.get(&member.estop).map(|key| IssuedKey {
            send: true,
            ..key.clone()
        })
    }

    /// Every key issued to generation `generation` of `participant` (with
    /// `send` where it is the sender), for its `GetKeys`: stream keys only
    /// where its device may read.
    pub fn get_keys(
        &self,
        participant: &ParticipantId,
        generation: u32,
        access: &Access,
    ) -> Vec<IssuedKey> {
        let Some(member) = self
            .members
            .get(participant)
            .filter(|m| m.generation == generation)
        else {
            return Vec::new();
        };
        let mut out: Vec<IssuedKey> = self
            .estop_key(participant, generation)
            .into_iter()
            .collect();
        let with =
            |id: &u32, send: bool| self.keys.get(id).map(|k| IssuedKey { send, ..k.clone() });
        if member.robot {
            let others = self.members.iter().filter(|(id, _)| *id != participant);
            out.extend(others.filter_map(|(_, m)| with(&m.estop, false)));
        }
        for (slot, state) in &self.slots {
            if &slot.robot == participant {
                if let Some(id) = state.sending
                    && state.sender_generation == generation
                {
                    out.extend(with(&id, true));
                }
            } else if access.may_read(&member.device, &slot.robot, &state.path) {
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
}
