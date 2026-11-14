//! Pending grants: their keys to the robots first, then commit or refuse.

use super::{
    Committed, DEVICE_PENDING, GRANT_WAIT_NS, GrantId, GrantRequest, KeyIssuer, LeaseKey, Pending,
    Prepared, ROBOT_PENDING,
};
use crate::crypto::Rng;
use crate::prelude::*;
use crate::wire::{
    ChannelId, CtlError, CtlStatus, IssuedKey, KeyPurpose, LeaseToken, ParticipantId, Security,
};
use alloc::collections::{BTreeMap, BTreeSet};

impl KeyIssuer {
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
            Security::Mac
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
            // A lockstep reply carries every part the robot publishes: only
            // a holder that may read them all gets one.
            if !request.reads_all.contains(robot) {
                continue;
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
