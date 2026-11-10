//! `ctl` requests: opening, dispatching and executing a verb, and its reply.

use std::sync::Arc;
use std::time::Duration;

use crate::wire::codec;
use crate::wire::{
    Command, CtlError, CtlReply, CtlStatus, DeviceView, IssuedKey, Lease, ParticipantId, ViewReply,
};
use rlmesh_session_core::auth::Admit;
use rlmesh_session_core::crypto::Kind;
use rlmesh_session_core::crypto::sealed::Header;
use tracing::{debug, info, warn};
use zenoh::query::Query;

use crate::authority::{Applied as Accepted, Identity};
#[cfg(feature = "fleet")]
use crate::host::actor::LobbyWaiting;
use crate::host::actor::Outcome;
use crate::host::actor::publish::asks_for_view;
use crate::host::peer::Peer;
use crate::host::{HostActor, lock};
#[cfg(feature = "fleet")]
use crate::lobby::is_lobby_verb;

/// What the host answers a request it cannot authenticate.
const UNSEALED: &str = "this session requires authenticated participants (feature auth.sealed): \
                        attach through the handshake with the session's address";

/// The least time between two Keeps of one session: each links the last
/// minutes of segments out of the purge's reach, so a flood would hold the
/// disk.
const KEEP_EVERY: Duration = Duration::from_secs(5);

/// The longest label a Keep may carry.
const KEEP_LABEL_MAX: usize = 64;

impl HostActor {
    /// Opens a `ctl` request under its sender's `k_up`.
    fn open_ctl(&self, mut raw: Vec<u8>) -> Result<(Arc<Peer>, Vec<u8>), CtlError> {
        let unauthenticated = |why: String| CtlError::new(CtlStatus::Unauthenticated, why);
        let header = Header::parse(&raw).map_err(|_| unauthenticated(UNSEALED.into()))?;
        let peer = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.by_key.get(&header.key_id).cloned())
            .ok_or_else(|| {
                unauthenticated(format!(
                    "key {} is unknown or superseded: attach again",
                    header.key_id
                ))
            })?;
        let body = lock(&peer.up)
            .open(&self.keys.ctl(), &mut raw)
            .map_err(|err| unauthenticated(format!("the request failed to open: {err}")))?
            .body
            .to_vec();
        Ok((peer, body))
    }

    pub(super) async fn on_ctl(&mut self, query: Query) {
        let raw = query
            .payload()
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
        let (peer, body) = match self.open_ctl(raw) {
            Ok(opened) => opened,
            Err(err) => {
                // Unauthenticated: a plain refusal, a hint only.
                info!(status = ?err.status, "ctl refused: {}", err.message);
                let reply = CtlReply {
                    result: Err(err),
                    state: self.snapshot(),
                    leases: Vec::new(),
                    request_id: 0,
                    keys: Vec::new(),
                    assignment: None,
                    run: None,
                };
                Self::reply(query, codec::encode_reply(&reply)).await;
                return;
            }
        };
        let request = codec::decode_request(&body);
        let request_id = request.as_ref().map_or(0, |r| r.request_id);
        let admitted = match lock(&peer.requests).admit(request_id) {
            Admit::Done(reply) => Some(reply.clone()),
            Admit::Expired => Some(Vec::new()),
            Admit::Pending => None,
            Admit::Execute => None,
        };
        // A retry of a lobby verb still waiting joins it.
        #[cfg(feature = "fleet")]
        let lobby_key = (
            peer.participant.clone(),
            peer.identity.generation,
            request_id,
        );
        #[cfg(feature = "fleet")]
        if admitted.is_none()
            && let Some(waiting) = self.lobby.get_mut(&lobby_key)
            && Arc::ptr_eq(&waiting.peer, &peer)
        {
            waiting.query = Some(query);
            return;
        }
        // A retry of a grant still waiting for its keys joins it.
        if admitted.is_none()
            && let Some(waiting) = self
                .waiting
                .values_mut()
                .find(|w| Arc::ptr_eq(&w.peer, &peer) && w.request.request_id == request_id)
        {
            waiting.queries.push(query);
            return;
        }
        let encoded = match admitted {
            // A retry of an answered request: the same reply, sealed afresh.
            Some(cached) if !cached.is_empty() => cached,
            Some(_) => self.answer(
                &peer.participant,
                Err(CtlError::new(
                    CtlStatus::Expired,
                    format!("request {request_id} is below the reply window: re-read the state"),
                )),
                Vec::new(),
                Vec::new(),
                request_id,
            ),
            None => {
                let outcome = match request {
                    Ok(request) => self.execute(&peer, request).await,
                    Err(err) => Outcome::refused(CtlError::new(CtlStatus::InvalidRequest, err)),
                };
                let (result, leases, keys) = match outcome {
                    // A verb a candidate may not send yet is dropped
                    // unanswered.
                    Outcome::Drop => return,
                    Outcome::Deferred(id) => {
                        lock(&peer.requests).begin(request_id);
                        if let Some(waiting) = self.waiting.get_mut(&id) {
                            waiting.queries.push(query);
                        }
                        return;
                    }
                    Outcome::Reply(result, leases, keys) => (result, leases, keys),
                    #[cfg(feature = "fleet")]
                    Outcome::Lobby(result) => {
                        let encoded = self.answer_lobby(&peer.participant, result, request_id);
                        lock(&peer.requests).complete(request_id, encoded.clone());
                        let sealed =
                            lock(&peer.down).seal(Kind::CtlReply, &self.keys.ctl(), &encoded);
                        match sealed {
                            Ok(sealed) => Self::reply(query, sealed).await,
                            Err(err) => {
                                warn!(participant = %peer.participant, %err, "reply not sealed")
                            }
                        }
                        return;
                    }
                    #[cfg(feature = "fleet")]
                    Outcome::Later(answer) => {
                        lock(&peer.requests).begin(request_id);
                        let done = self.lobby_done.clone();
                        let key = lobby_key.clone();
                        let task = tokio::spawn(async move {
                            let result = answer.await.unwrap_or_else(|_| {
                                Err(CtlError::new(
                                    CtlStatus::Expired,
                                    "the lobby dropped this request: send it again",
                                ))
                            });
                            let _ = done.send((key, result));
                        });
                        self.lobby.insert(
                            lobby_key,
                            LobbyWaiting {
                                peer: peer.clone(),
                                query: Some(query),
                                task: task.abort_handle(),
                            },
                        );
                        return;
                    }
                };
                let encoded = self.answer(&peer.participant, result, leases, keys, request_id);
                lock(&peer.requests).complete(request_id, encoded.clone());
                encoded
            }
        };
        // Ticks go out before the reply: a `Ready` that completed a step
        // returns after the step is published.
        self.publish_ticks().await;
        let sealed = lock(&peer.down).seal(Kind::CtlReply, &self.keys.ctl(), &encoded);
        match sealed {
            Ok(sealed) => Self::reply(query, sealed).await,
            Err(err) => warn!(participant = %peer.participant, %err, "reply not sealed"),
        }
    }

    /// Applies a verb from `peer`.
    async fn execute(&mut self, peer: &Arc<Peer>, request: crate::wire::Request) -> Outcome {
        let now = self.now();
        if request.from != peer.participant {
            return Outcome::refused(CtlError::new(
                CtlStatus::Unauthenticated,
                format!(
                    "sealed under {}'s keys but sent as {}",
                    peer.participant, request.from
                ),
            ));
        }
        let in_effect = self
            .peers
            .read()
            .ok()
            .and_then(|p| p.current.get(&peer.participant).cloned())
            .is_some_and(|current| Arc::ptr_eq(&current, peer));
        if !in_effect {
            if !matches!(request.command, Command::Attach { .. }) {
                return Outcome::Drop;
            }
            let Identity { device, generation } = &peer.identity;
            if let Err(err) = self
                .auth
                .may_take_effect(&peer.participant, device, *generation)
            {
                return Outcome::refused(err);
            }
        }
        debug!(from = %request.from, command = ?request.command, "ctl");
        #[cfg(feature = "fleet")]
        if is_lobby_verb(&request.command)
            && let Some(lobby) = self.config.lobby.clone()
        {
            return self.lobby_verb(peer, lobby.as_ref(), request.command);
        }
        let generation = peer.identity.generation;
        match &request.command {
            Command::KeysAck(ack) => {
                self.auth
                    .issuer()
                    .ack(&peer.participant, generation, ack, now);
            }
            Command::GetKeys => {
                let keys = self.auth.issuer().get_keys(
                    &peer.participant,
                    generation,
                    &self.authority.config().access,
                );
                return Outcome::Reply(Ok(()), Vec::new(), keys);
            }
            Command::GrantLease { .. } | Command::DeclareBinding { .. } => {
                return self.mint(peer, request).await;
            }
            Command::Keep { .. } => return self.keep(peer, request, now),
            Command::DeclareParts { board_id, .. } => {
                if let Err(err) = self.auth.check_board(
                    &peer.participant,
                    &peer.identity.device,
                    board_id.as_ref(),
                ) {
                    info!(status = ?err.status, "ctl refused: {}", err.message);
                    return Outcome::refused(err);
                }
            }
            _ => {}
        }
        let attaching = matches!(request.command, Command::Attach { .. });
        let result = self.authority.apply_from(request, &peer.identity, now);
        match result {
            Ok(Accepted { leases, events }) => {
                if !in_effect {
                    self.take_effect(peer);
                }
                self.record(events).await;
                // An attach's reply carries the generation's `k_estop`.
                let keys = if attaching {
                    let estop = self.auth.issuer().estop_key(&peer.participant, generation);
                    estop.into_iter().collect()
                } else {
                    Vec::new()
                };
                Outcome::Reply(Ok(()), leases, keys)
            }
            Err(err) => {
                info!(status = ?err.status, "ctl refused: {}", err.message);
                Outcome::refused(err)
            }
        }
    }

    /// `Keep`: the authority checks the sender, then the recording is kept.
    fn keep(&mut self, peer: &Arc<Peer>, request: crate::wire::Request, now: i64) -> Outcome {
        let Command::Keep { label } = &request.command else {
            return Outcome::Drop;
        };
        let Some(keep) = self.config.keep.clone() else {
            return Outcome::refused(CtlError::new(
                CtlStatus::Unsupported,
                "this host records nothing to keep",
            ));
        };
        if label.len() > KEEP_LABEL_MAX {
            return Outcome::refused(CtlError::new(
                CtlStatus::InvalidRequest,
                format!("a Keep label has at most {KEEP_LABEL_MAX} bytes"),
            ));
        }
        let every = i64::try_from(KEEP_EVERY.as_nanos()).unwrap_or(i64::MAX);
        if self
            .last_keep
            .is_some_and(|t| now.saturating_sub(t) < every)
        {
            return Outcome::refused(CtlError::new(
                CtlStatus::NotReady,
                format!("kept less than {} s ago", KEEP_EVERY.as_secs()),
            ));
        }
        let label = label.clone();
        if let Err(err) = self.authority.apply_from(request, &peer.identity, now) {
            info!(status = ?err.status, "ctl refused: {}", err.message);
            return Outcome::refused(err);
        }
        self.last_keep = Some(now);
        match keep(self.config.session, &label) {
            Ok(dir) => {
                info!(by = %peer.participant, %dir, "recording kept");
                Outcome::Reply(Ok(()), Vec::new(), Vec::new())
            }
            Err(why) => {
                warn!(by = %peer.participant, %why, "recording not kept");
                Outcome::refused(CtlError::new(CtlStatus::NotReady, why))
            }
        }
    }

    /// The encoded reply: the result, the snapshot right after it and the
    /// caller's keys.
    pub(super) fn answer(
        &self,
        to: &ParticipantId,
        result: Result<(), CtlError>,
        leases: Vec<Lease>,
        keys: Vec<IssuedKey>,
        request_id: u64,
    ) -> Vec<u8> {
        self.answer_with(to, result, (leases, keys), request_id, (None, None))
    }

    pub(super) fn answer_with(
        &self,
        to: &ParticipantId,
        result: Result<(), CtlError>,
        (leases, keys): (Vec<Lease>, Vec<IssuedKey>),
        request_id: u64,
        (assignment, run): (
            Option<crate::wire::Assignment>,
            Option<crate::wire::RunState>,
        ),
    ) -> Vec<u8> {
        let state = self.snapshot();
        // A robot that takes its device view gets it in place of the state: a board
        // never decodes the state.
        if asks_for_view(&state, to) {
            return codec::encode_view_reply(&ViewReply {
                result,
                view: Some(DeviceView::of(&state, to)),
                leases,
                request_id,
                keys,
                assignment,
                run,
            });
        }
        codec::encode_reply(&CtlReply {
            result,
            state,
            leases,
            request_id,
            keys,
            assignment,
            run,
        })
    }
}
