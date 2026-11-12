//! Serving on a session: the sessions a model joins at its drivers'
//! invitation, and the loop answering its requests on `m/{model}`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rlmesh_session::address::SessionAddress;
use rlmesh_session::transport::{Transport, payload_bytes};
use rlmesh_session::wire::codec::{decode_model_request, encode_model_reply};
use rlmesh_session::wire::timing::ns;
use rlmesh_session::wire::{
    ModelFault, ModelReply, ModelRequest, ParticipantId, PriorityClass, Role, SessionId, TenantId,
};
use rlmesh_session::{Participant, SessionError};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::{ModelService, PROBE, fault};
use crate::lockstep::error::GymError;
use crate::lockstep::session::lockstep_transport;
use crate::model::handler::ModelHandler;

/// The sessions a served model joined at its drivers' invitation, each
/// answered with the same service.
pub(super) struct Joins<H> {
    pub(super) name: ParticipantId,
    pub(super) service: Arc<ModelService<H>>,
    pub(super) timeout: Duration,
    pub(super) shutdown: CancellationToken,
    /// The sessions joined, being joined or being left.
    pub(super) sessions: std::sync::Mutex<HashMap<SessionKey, Joined>>,
}

/// A session as its host knows it, however its address is spelled.
type SessionKey = (std::net::SocketAddr, TenantId, SessionId);

fn session_key(address: &SessionAddress) -> SessionKey {
    (address.base, address.tenant.clone(), address.session)
}

/// One invited session: its inviters still in it, what stops answering
/// there, the first inviter's join (`None` while it runs), which the others
/// wait on rather than joining again under the same name, and its end. It
/// stays until the model has left the session, so an invitation while it
/// leaves waits for that instead of joining beside it.
pub(super) struct Joined {
    inviters: usize,
    cancel: CancellationToken,
    joined: watch::Receiver<Option<Result<(), ModelFault>>>,
    left: CancellationToken,
}

/// What an invitation does about the session's entry.
enum Invited {
    /// Joins it, the first inviter.
    Join(
        CancellationToken,
        watch::Sender<Option<Result<(), ModelFault>>>,
    ),
    /// Waits on the first inviter's join.
    Follow(watch::Receiver<Option<Result<(), ModelFault>>>),
    /// Waits for the model to have left it, then joins again.
    Wait(CancellationToken),
}

impl<H: ModelHandler + 'static> Joins<H> {
    fn sessions(&self) -> std::sync::MutexGuard<'_, HashMap<SessionKey, Joined>> {
        self.sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Forgets the session at `key`, which the model has left (or never
    /// joined).
    fn forget(&self, key: &SessionKey) {
        if let Some(joined) = self.sessions().remove(key) {
            joined.left.cancel();
        }
    }

    /// Joins the session at `text` (once per session), answering there
    /// until its inviters leave or nobody else is left.
    async fn join(self: &Arc<Self>, text: &str) -> Result<(), ModelFault> {
        let address: SessionAddress = text.parse().map_err(|e: SessionError| {
            // A host name that does not resolve yet is out of reach, not
            // a malformed request.
            let code = match e {
                SessionError::Io(_) => "UNAVAILABLE",
                _ => "INVALID_REQUEST",
            };
            fault(code, e.to_string(), false)
        })?;
        let key = session_key(&address);
        loop {
            if self.shutdown.is_cancelled() {
                return Err(fault("UNAVAILABLE", "the model is closing", false));
            }
            // One lock decides: join here, wait on the join in progress, or
            // wait for the last one to have left.
            let invited = {
                let mut sessions = self.sessions();
                match sessions.get_mut(&key) {
                    Some(joined) if joined.cancel.is_cancelled() => {
                        Invited::Wait(joined.left.clone())
                    }
                    Some(joined) => {
                        joined.inviters += 1;
                        Invited::Follow(joined.joined.clone())
                    }
                    None => {
                        let cancel = self.shutdown.child_token();
                        let (done, joined) = watch::channel(None);
                        sessions.insert(
                            key.clone(),
                            Joined {
                                inviters: 1,
                                cancel: cancel.clone(),
                                joined,
                                left: CancellationToken::new(),
                            },
                        );
                        Invited::Join(cancel, done)
                    }
                }
            };
            match invited {
                Invited::Wait(left) => {
                    #[cfg(test)]
                    leaving::waits(&self.name);
                    left.cancelled().await;
                }
                Invited::Follow(mut outcome) => {
                    return match outcome.wait_for(Option::is_some).await {
                        Ok(outcome) => outcome.clone().unwrap_or(Ok(())),
                        Err(_) => Err(fault(
                            "UNAVAILABLE",
                            "the model's join was abandoned",
                            false,
                        )),
                    };
                }
                Invited::Join(cancel, done) => {
                    let outcome = self.answer_there(address, key, cancel).await;
                    done.send_replace(Some(outcome.clone()));
                    return outcome;
                }
            }
        }
    }

    /// Joins the session at `address` and answers there until `cancel`
    /// fires, forgetting it once the model has left.
    async fn answer_there(
        self: &Arc<Self>,
        address: SessionAddress,
        key: SessionKey,
        cancel: CancellationToken,
    ) -> Result<(), ModelFault> {
        let joined = address
            .join_declaring(
                self.name.as_str(),
                vec![Role::Controller],
                PriorityClass::Model,
                self.timeout,
                self.service.editions.clone(),
            )
            .await;
        let participant = match joined {
            Ok(participant) => participant,
            Err(e) => {
                self.forget(&key);
                return Err(fault(
                    "UNAVAILABLE",
                    format!(
                        "model {} can't reach the session at {address}: {e}; run `rlmesh host` \
                         somewhere both machines can reach, pass `session=`, or serve the env \
                         at an address its host can reach",
                        self.name
                    ),
                    false,
                ));
            }
        };
        // Alone in the session: every inviter is gone, whether or not it
        // said so. Its inviters join after this answers, so it counts as
        // alone only once someone else was there, or when nobody came
        // within the join timeout.
        let mut roster = participant.watch();
        let (alone, me) = (cancel.clone(), self.name.clone());
        let deadline = tokio::time::Instant::now() + self.timeout;
        tokio::spawn(async move {
            let mut seen = false;
            loop {
                let only_me = roster
                    .borrow_and_update()
                    .as_ref()
                    .map(|s| s.roster.iter().all(|p| p.id == me));
                match only_me {
                    Some(true) if seen => break,
                    Some(false) => seen = true,
                    _ => {}
                }
                let changed = if seen {
                    roster.changed().await
                } else {
                    match tokio::time::timeout_at(deadline, roster.changed()).await {
                        Ok(changed) => changed,
                        Err(_) => break,
                    }
                };
                if changed.is_err() {
                    return;
                }
            }
            alone.cancel();
        });
        let joins = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(err) = answer_in(joins.clone(), participant, address.clone(), cancel).await {
                warn!(%err, session = %address, "model left an invited session");
            }
            #[cfg(test)]
            leaving::left(&joins.name).await;
            joins.forget(&key);
        });
        Ok(())
    }

    /// One inviter of the session at `text` left; the last one's leaving
    /// stops the model answering there.
    fn leave(&self, text: &str) {
        let Ok(address) = text.parse::<SessionAddress>() else {
            return;
        };
        if let Some(joined) = self.sessions().get_mut(&session_key(&address)) {
            joined.inviters = joined.inviters.saturating_sub(1);
            if joined.inviters == 0 {
                joined.cancel.cancel();
            }
        }
    }
}

/// Answers on `participant`'s session (at `session`) until `cancel`
/// fires, then leaves it. Boxed: an invitation answered there serves the
/// invited session with this same future.
pub(super) fn answer_in<H: ModelHandler + 'static>(
    joins: Arc<Joins<H>>,
    participant: Participant,
    session: SessionAddress,
    cancel: CancellationToken,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), GymError>> + Send>> {
    Box::pin(async move {
        let lockstep = lockstep_transport(participant.transport(), &session).await?;
        let key = participant.keys().model(&joins.name);
        info!(model = %joins.name, %key, "model serving");
        let served = serve_zenoh(lockstep.clone(), key, joins, cancel).await;
        if crate::lockstep::session::owns_links(participant.transport()) {
            let _ = lockstep.close().await;
        }
        participant.close().await;
        served
    })
}

/// Answers the model's requests on `key` until `cancel` fires. Requests run
/// side by side; the handler's lock orders forwards.
async fn serve_zenoh<H: ModelHandler + 'static>(
    session: Transport,
    key: String,
    joins: Arc<Joins<H>>,
    cancel: CancellationToken,
) -> Result<(), GymError> {
    let queryable = session
        .control()
        .declare_queryable(&key)
        .complete(true)
        .await
        .map_err(|e| GymError::Closed(e.to_string()))?;
    loop {
        let query = tokio::select! {
            () = cancel.cancelled() => break,
            query = queryable.recv_async() => match query {
                Ok(query) => query,
                Err(_) => break,
            },
        };
        let (joins, key) = (joins.clone(), key.clone());
        tokio::spawn(async move {
            let arrived = Instant::now();
            let (reply, timing) = if query.parameters().contains_key(PROBE) {
                (ModelReply::Done, None)
            } else {
                let payload = query.payload().map(payload_bytes).unwrap_or_default();
                match decode_model_request(payload) {
                    Ok(ModelRequest::Join { session }) => {
                        let joined = joins.join(&session).await;
                        (
                            joined.map_or_else(ModelReply::Fault, |()| ModelReply::Done),
                            None,
                        )
                    }
                    Ok(ModelRequest::Leave { session }) => {
                        joins.leave(&session);
                        (ModelReply::Done, None)
                    }
                    Ok(request) => {
                        let service = &joins.service;
                        let decoded = arrived.elapsed();
                        let (reply, mut timing) = service.answer(request).await;
                        timing.decode_ns =
                            ns(decoded).map(|wire| wire + timing.decode_ns.unwrap_or(0));
                        timing.total_ns = ns(arrived.elapsed());
                        (reply, Some(timing))
                    }
                    Err(err) => (
                        ModelReply::Fault(fault("INVALID_REQUEST", err.to_string(), false)),
                        None,
                    ),
                }
            };
            if let Err(err) = query
                .reply(&key, encode_model_reply(&reply, timing.as_ref()))
                .express(true)
                .await
            {
                warn!(%err, "model reply not sent");
            }
        });
    }
    Ok(())
}

/// Holds a model's leaving of an invited session, once, until an invitation
/// waits on it, so a test can make the two meet.
#[cfg(test)]
mod leaving {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    use rlmesh_session::wire::ParticipantId;
    use tokio::sync::Notify;

    /// Notified as the model leaves, and as an invitation waits on that.
    #[derive(Default)]
    pub(super) struct Hold {
        pub(super) leaving: Notify,
        waits: Notify,
    }

    fn holds() -> std::sync::MutexGuard<'static, HashMap<String, Arc<Hold>>> {
        static HOLDS: OnceLock<Mutex<HashMap<String, Arc<Hold>>>> = OnceLock::new();
        HOLDS
            .get_or_init(Mutex::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Holds the next leaving of the model named `model`.
    pub(super) fn hold(model: &str) -> Arc<Hold> {
        Arc::clone(holds().entry(model.to_string()).or_default())
    }

    pub(super) async fn left(model: &ParticipantId) {
        let hold = holds().get(model.as_str()).cloned();
        if let Some(hold) = hold {
            hold.leaving.notify_one();
            hold.waits.notified().await;
            holds().remove(model.as_str());
        }
    }

    pub(super) fn waits(model: &ParticipantId) {
        if let Some(hold) = holds().get(model.as_str()) {
            hold.waits.notify_one();
        }
    }
}

#[cfg(test)]
mod rejoin_tests {
    use super::*;
    use crate::lockstep::model::{ModelBinding, ModelConnection};

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_invitation_while_the_model_leaves_joins_once_it_has_left() {
        let binding = ModelBinding::bind(":0/rejoin").await.unwrap();
        let address = binding.address();
        let cancel = CancellationToken::new();
        let service = Arc::new(ModelService::new(
            crate::model::random::RandomSampleHandler::default(),
            String::new(),
        ));
        let served = tokio::spawn(binding.serve(service, Duration::from_secs(2), cancel.clone()));
        let mut target = rlmesh_session::address::host_at(
            "127.0.0.1:0".parse().unwrap(),
            "bench",
            Duration::from_secs(1),
            false,
        )
        .await
        .unwrap();
        let session = target.address.invite();
        let invite = |id: &'static str| {
            let (address, session) = (address.clone(), session.clone());
            async move {
                ModelConnection::connect(&address, id, Duration::from_secs(5))
                    .await
                    .unwrap()
                    .invite(&session, id, Duration::from_secs(5))
                    .await
            }
        };

        let driver = invite("driver").await.unwrap();
        let hold = leaving::hold("rejoin");
        driver.close().await;
        hold.leaving.notified().await;
        // The model is still leaving: the invitation waits for that, then
        // joins again and answers.
        invite("driver-again").await.unwrap().close().await;
        cancel.cancel();
        served.await.unwrap().unwrap();
        target.close().await.unwrap();
    }
}

#[cfg(test)]
mod blocking_tests {
    use super::*;
    use crate::lockstep::linked_host::{LinkedHost, timer_survives};

    #[test]
    fn dropping_a_model_answering_in_a_session_takes_no_wakeup_of_another_future() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (host, answering) = runtime.block_on(async {
            let host = LinkedHost::start().await;
            let participant = host
                .attach("model", Role::Controller, PriorityClass::Model)
                .await;
            assert!(crate::lockstep::session::owns_links(
                participant.transport()
            ));
            let name = ParticipantId::new("model").unwrap();
            let key = host.keys().model(&name);
            let joins = Arc::new(Joins {
                name,
                service: Arc::new(ModelService::new(
                    crate::model::random::RandomSampleHandler::default(),
                    String::new(),
                )),
                timeout: Duration::from_secs(5),
                shutdown: CancellationToken::new(),
                sessions: std::sync::Mutex::default(),
            });
            let mut answering = answer_in(
                joins,
                participant,
                host.address.clone(),
                CancellationToken::new(),
            );
            // Answering over its own lockstep session once that answers.
            tokio::select! {
                ended = &mut answering => panic!("the model stopped answering: {ended:?}"),
                () = host.answered(&key) => {}
            }
            (host, answering)
        });
        // The answer is cancelled: its lockstep session goes unclosed.
        let dropped = timer_survives(async move { drop(answering) });
        assert!(
            dropped,
            "a cancelled model's lockstep session took a pending wakeup"
        );
        drop(host);
    }
}
