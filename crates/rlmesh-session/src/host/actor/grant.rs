//! Grants that mint a token: prepared, held until the robots acknowledge their keys, then committed.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::wire::{ChannelId, CtlError, CtlStatus, IssuedKey, Lease, ParticipantId};
use rlmesh_session_core::auth::issuer::{GrantId, GrantPart, GrantRequest, Prepared};
use rlmesh_session_core::crypto::Kind;
use tracing::{info, warn};

use crate::host::actor::{Outcome, Waiting};
use crate::host::peer::Peer;
use crate::host::{HostActor, OsRng, lock};

impl HostActor {
    /// A verb that may mint a lease token (a grant, a rebind): tried on a
    /// copy of the authority with a reserved token. Minting nothing, it is
    /// applied at once; otherwise its keys go to its robots and it waits.
    pub(super) async fn mint(
        &mut self,
        peer: &Arc<Peer>,
        request: crate::wire::Request,
    ) -> Outcome {
        let now = self.now();
        let token = self.authority.reserve_token();
        let mut trial = self.authority.clone();
        let applied = match trial.apply_minting(request.clone(), &peer.identity, now, token) {
            Ok(applied) => applied,
            Err(err) => {
                info!(status = ?err.status, "ctl refused: {}", err.message);
                return Outcome::refused(err);
            }
        };
        let minted: Vec<Lease> = trial
            .leases()
            .filter(|l| l.token == token)
            .cloned()
            .collect();
        if minted.is_empty() {
            self.authority = trial;
            self.record(applied.events).await;
            return Outcome::Reply(Ok(()), applied.leases, Vec::new());
        }
        let parts: Vec<GrantPart> = minted
            .iter()
            .filter_map(|lease| {
                let part = self.authority.part(lease.channel)?;
                Some(GrantPart {
                    robot: part.owner.clone(),
                    channel: lease.channel,
                    priority: part.priority,
                    security: self.authority.security(lease.channel),
                })
            })
            .collect();
        let mut robots: Vec<(ParticipantId, u32)> = Vec::new();
        for part in &parts {
            if robots.iter().all(|(r, _)| r != &part.robot) {
                let generation = self
                    .authority
                    .participant(&part.robot)
                    .map_or(0, |p| p.generation);
                robots.push((part.robot.clone(), generation));
            }
        }
        let access = &self.authority.config().access;
        let reads_all = robots
            .iter()
            .map(|(robot, _)| robot)
            .filter(|robot| {
                self.authority
                    .parts()
                    .filter(|p| &p.owner == *robot)
                    .all(|p| access.may_read(&peer.identity.device, robot, p.path.as_str()))
            })
            .cloned()
            .collect();
        let grant = GrantRequest {
            holder: peer.participant.clone(),
            holder_generation: peer.identity.generation,
            device: peer.identity.device.clone(),
            class: minted[0].priority,
            token,
            parts,
            reads_all,
        };
        match self.auth.issuer().prepare_grant(&mut OsRng, grant, now) {
            Ok(Prepared { id, evicted }) => {
                for victim in evicted {
                    let preempted = CtlError::new(
                        CtlStatus::LeaseBusy,
                        "preempted: a higher class's grant took this robot's last pending slot",
                    );
                    self.finish_grant(victim, Err(preempted)).await;
                }
                self.waiting.insert(
                    id,
                    Waiting {
                        peer: peer.clone(),
                        request,
                        token,
                        epoch: self.authority.epoch(),
                        robots,
                        queries: Vec::new(),
                    },
                );
                Outcome::Deferred(id)
            }
            Err(err) => {
                info!(status = ?err.status, "grant refused: {}", err.message);
                Outcome::refused(err)
            }
        }
    }

    /// Grant `id`'s robots all installed its keys: re-check it against the
    /// current state (the holder's and robots' generations, the epoch, and
    /// every grant check) and commit it, or refuse it.
    pub(super) async fn commit_grant(
        &mut self,
        id: GrantId,
    ) -> Result<(Vec<Lease>, Vec<IssuedKey>), CtlError> {
        let changed = |why: &str| {
            CtlError::new(
                CtlStatus::StaleEpoch,
                format!("{why} while the lease's keys were delivered"),
            )
        };
        let Some(waiting) = self.waiting.get(&id) else {
            return Err(changed("the grant ended"));
        };
        let holder_current = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(&waiting.peer.participant).cloned())
            .is_some_and(|current| Arc::ptr_eq(&current, &waiting.peer));
        let robots_current = waiting.robots.iter().all(|(robot, generation)| {
            self.authority
                .participant(robot)
                .is_some_and(|p| p.generation == *generation)
        });
        if !holder_current || !robots_current {
            self.auth.issuer().refuse_grant(id);
            return Err(changed("a holder or robot restarted"));
        }
        if waiting.epoch != self.authority.epoch() {
            self.auth.issuer().refuse_grant(id);
            return Err(changed("the epoch changed"));
        }
        let (request, identity, token) = (
            waiting.request.clone(),
            waiting.peer.identity.clone(),
            waiting.token,
        );
        let mut trial = self.authority.clone();
        let applied = match trial.apply_minting(request, &identity, self.now(), token) {
            Ok(applied) => applied,
            Err(err) => {
                self.auth.issuer().refuse_grant(id);
                return Err(err);
            }
        };
        let minted: BTreeMap<ChannelId, ()> = trial
            .leases()
            .filter(|l| l.token == token)
            .map(|l| (l.channel, ()))
            .collect();
        let prepared: BTreeMap<ChannelId, ()> = self
            .auth
            .issuer()
            .grant(id)
            .map(|g| g.parts.iter().map(|p| (p.channel, ())).collect())
            .unwrap_or_default();
        if minted != prepared {
            self.auth.issuer().refuse_grant(id);
            return Err(changed("the parts it covers changed"));
        }
        let Some(committed) = self.auth.issuer().commit_grant(id) else {
            return Err(changed("the grant ended"));
        };
        for (channel, key_id) in &committed.key_ids {
            trial.set_key_id(*channel, *key_id);
        }
        let leases = applied
            .leases
            .iter()
            .map(|l| trial.lease(l.channel).cloned().unwrap_or_else(|| l.clone()))
            .collect();
        self.authority = trial;
        self.record(applied.events).await;
        Ok((leases, committed.keys))
    }

    /// Answers every query waiting on grant `id`, once.
    pub(super) async fn finish_grant(
        &mut self,
        id: GrantId,
        outcome: Result<(Vec<Lease>, Vec<IssuedKey>), CtlError>,
    ) {
        let Some(waiting) = self.waiting.remove(&id) else {
            return;
        };
        let request_id = waiting.request.request_id;
        let to = &waiting.peer.participant;
        let encoded = match outcome {
            Ok((leases, keys)) => self.answer(to, Ok(()), leases, keys, request_id),
            Err(err) => {
                info!(status = ?err.status, "grant refused: {}", err.message);
                self.answer(to, Err(err), Vec::new(), Vec::new(), request_id)
            }
        };
        lock(&waiting.peer.requests).complete(request_id, encoded.clone());
        let key = self.keys.ctl();
        for query in waiting.queries {
            let sealed = lock(&waiting.peer.down).seal(Kind::CtlReply, &key, &encoded);
            match sealed {
                Ok(sealed) => Self::reply(query, sealed),
                Err(err) => warn!(%err, "grant reply not sealed"),
            }
        }
    }
}
