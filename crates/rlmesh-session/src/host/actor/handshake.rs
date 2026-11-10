//! Handshakes: a candidate's keys, and the generation taking effect on its `Attach`.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::wire::codec;
use crate::wire::{CtlError, CtlStatus, Role};
use rlmesh_session_core::auth::issuer::Joining;
use tracing::{debug, info, warn};
use zenoh::query::Query;

use crate::authority::Identity;
use crate::host::peer::{Peer, Peers};
use crate::host::{HostActor, OsRng};

/// How long a handshake's keys wait for their `Attach` before the next
/// handshake drops them.
const CANDIDATE_TTL: Duration = Duration::from_secs(10);

impl HostActor {
    /// Answers msg 1 of a handshake: msg 2 and a candidate, or a refusal.
    pub(super) async fn on_handshake(&mut self, query: Query) {
        let now = self.now();
        let answer = query
            .payload()
            .ok_or_else(|| CtlError::new(CtlStatus::InvalidRequest, "a handshake without msg 1"))
            .and_then(|payload| {
                codec::decode_handshake_request(&payload.to_bytes())
                    .map_err(|err| CtlError::new(CtlStatus::InvalidRequest, err))
            })
            .and_then(|request| {
                self.auth.handshake(
                    &mut OsRng,
                    &self.config.identity,
                    &request,
                    &self.authority.config().access,
                )
            });
        let reply = match answer {
            Ok((msg2, candidate)) => {
                debug!(participant = %candidate.participant, generation = candidate.generation, "handshake");
                let ttl = i64::try_from(CANDIDATE_TTL.as_nanos()).unwrap_or(i64::MAX);
                let peer = Arc::new(Peer {
                    participant: candidate.participant,
                    identity: Identity {
                        device: candidate.device,
                        generation: candidate.generation,
                    },
                    created_ns: now,
                    up: Mutex::new(candidate.up),
                    down: Mutex::new(candidate.down),
                    requests: Mutex::default(),
                    echo: std::sync::OnceLock::new(),
                    report: Mutex::default(),
                });
                if let Ok(mut peers) = self.peers.write() {
                    let Peers { by_key, current } = &mut *peers;
                    by_key.retain(|_, p| {
                        current
                            .get(&p.participant)
                            .is_some_and(|c| Arc::ptr_eq(c, p))
                            || now.saturating_sub(p.created_ns) < ttl
                    });
                    by_key.insert(candidate.up_key_id, peer);
                }
                Ok(msg2)
            }
            Err(err) => {
                info!(status = ?err.status, "handshake refused: {}", err.message);
                Err(err)
            }
        };
        Self::reply(query, codec::encode_handshake_reply(&reply)).await;
    }

    /// `peer`'s `Attach` was accepted: it is the generation in effect, and
    /// every older one of its participant is dropped. It gets the stream
    /// keys it reads, and a robot's streams rotate to it.
    pub(super) fn take_effect(&mut self, peer: &Arc<Peer>) {
        let Identity { device, generation } = &peer.identity;
        self.auth
            .take_effect(&peer.participant, device, *generation);
        let now = self.now();
        let robot = self
            .authority
            .participant(&peer.participant)
            .is_some_and(|p| p.has_role(Role::Embodiment));
        let joining = Joining {
            participant: &peer.participant,
            generation: *generation,
            device,
            robot,
        };
        match self
            .auth
            .issuer()
            .attach(&mut OsRng, joining, &self.authority.config().access, now)
        {
            Ok(failed) => self.failed.extend(failed),
            Err(err) => warn!(participant = %peer.participant, %err, "keys"),
        }
        if let Ok(mut peers) = self.peers.write() {
            peers.by_key.retain(|_, p| {
                p.participant != peer.participant || p.identity.generation >= *generation
            });
            peers.current.insert(peer.participant.clone(), peer.clone());
        }
    }
}
