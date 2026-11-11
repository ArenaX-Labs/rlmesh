//! The driver side's way to a served model ([`ModelLink`]), a served model's
//! placed address ([`ModelBinding`]), and a driver's connection to it.

use std::sync::Arc;
use std::time::Duration;

use rlmesh_session::address::{Address, LEASE_GRACE, Placed, SessionAddress, place};
use rlmesh_session::transport::payload_bytes;
use rlmesh_session::wire::codec::{decode_model_reply, encode_model_request};
use rlmesh_session::wire::{
    EndpointTiming, ModelReply, ModelRequest, ParticipantId, PriorityClass, Role,
};
use rlmesh_session::{Participant, SessionError};
use tokio_util::sync::CancellationToken;
use zenoh::qos::{CongestionControl, Priority};
use zenoh::query::ConsolidationMode;

use super::joins::{Joins, answer_in};
use super::{ModelService, PROBE};
use crate::lockstep::error::GymError;
use crate::lockstep::session::lockstep_session;
use crate::model::handler::ModelHandler;

/// How long a model query waits when the request sets no deadline.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// The driver side's way to a served model: queries on `m/{model}`.
#[derive(Clone)]
pub struct ModelLink {
    session: zenoh::Session,
    key: String,
}

impl ModelLink {
    /// A link to the model answering on `key`.
    pub fn new(session: zenoh::Session, key: String) -> Self {
        Self { session, key }
    }

    /// Waits until the model answers, probing every 20 ms for up to
    /// `timeout`.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), GymError> {
        let deadline = tokio::time::Instant::now() + timeout;
        let selector = format!("{}?{PROBE}", self.key);
        loop {
            let replies = self
                .session
                .get(&selector)
                .consolidation(ConsolidationMode::None)
                .timeout(Duration::from_millis(500))
                .await
                .map_err(|e| GymError::Closed(e.to_string()))?;
            if let Ok(reply) = replies.recv_async().await
                && reply.result().is_ok()
            {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(GymError::Timeout(timeout));
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// Sends one request and waits for its reply and what the model
    /// measured answering it.
    pub async fn call(
        &self,
        request: &ModelRequest,
        timeout: Option<Duration>,
    ) -> Result<(ModelReply, Option<EndpointTiming>), GymError> {
        let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
        let started = tokio::time::Instant::now();
        let replies = self
            .session
            .get(&self.key)
            .payload(encode_model_request(request))
            .consolidation(ConsolidationMode::None)
            .priority(Priority::RealTime)
            .congestion_control(CongestionControl::Block)
            .express(true)
            .timeout(timeout)
            .await
            .map_err(|e| GymError::Closed(e.to_string()))?;
        let reply = replies.recv_async().await.map_err(|_| {
            // Replies that end with none before the timeout: the model's
            // server went away while the request was in flight.
            if started.elapsed() + crate::lockstep::exchange::ENDED_EARLY < timeout {
                GymError::Closed(
                    "the served model went away before it answered: its server exited or left \
                     the session"
                        .to_owned(),
                )
            } else {
                GymError::Timeout(timeout)
            }
        })?;
        match reply.result() {
            Ok(sample) => Ok(decode_model_reply(payload_bytes(sample.payload()))?),
            // A Zenoh-level error (the queryable is gone).
            Err(_) => Err(GymError::Closed(format!("{} did not answer", self.key))),
        }
    }
}

/// A served model's participant address, its session placed.
pub struct ModelBinding {
    session: SessionAddress,
    name: ParticipantId,
    placed: Placed,
}

/// The model's address without the key, for logs.
impl std::fmt::Display for ModelBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.session, self.name)
    }
}

impl ModelBinding {
    /// Places the session of the participant address `address` (the
    /// model's name is its participant) by the hosting rule.
    pub async fn bind(address: &str) -> Result<Self, GymError> {
        let parsed = Address::participant(address).map_err(|e| GymError::Config(e.to_string()))?;
        let name = ParticipantId::new(parsed.participant.clone().unwrap_or_default())
            .map_err(|e| GymError::Config(e.to_string()))?;
        let placed = place(&parsed, LEASE_GRACE).await?;
        Ok(Self {
            session: placed.address.clone(),
            name,
            placed,
        })
    }

    /// The full address drivers reach the model at: an invite, key
    /// included ([`SessionAddress::invite`]).
    pub fn address(&self) -> String {
        self.session.participant_invite(self.name.as_str())
    }

    /// Joins the session as the model and answers `service` until
    /// `shutdown` fires, then closes it, waiting up to `timeout` (also the
    /// join's) for its accepted requests; the session's host stops either
    /// way.
    pub async fn serve<H: ModelHandler + 'static>(
        self,
        service: Arc<ModelService<H>>,
        timeout: Duration,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        let result = self.serve_inner(&service, timeout, shutdown).await;
        let closed = service.close(timeout).await;
        let mut placed = self.placed;
        let _ = placed.close().await;
        result?;
        closed.map_err(|e| GymError::Config(e.to_string()))
    }

    async fn serve_inner<H: ModelHandler + 'static>(
        &self,
        service: &Arc<ModelService<H>>,
        timeout: Duration,
        shutdown: CancellationToken,
    ) -> Result<(), GymError> {
        // A model drives nothing itself: it joins as a controller at model
        // priority, so the roster shows it, and never takes a lease.
        let participant: Participant = self
            .session
            .join_declaring(
                self.name.as_str(),
                vec![Role::Controller],
                PriorityClass::Model,
                timeout,
                service.editions.clone(),
            )
            .await?;
        service.serving_as(self.name.as_str(), self.address());
        let joins = Arc::new(Joins {
            name: self.name.clone(),
            service: Arc::clone(service),
            timeout,
            shutdown: shutdown.clone(),
            sessions: std::sync::Mutex::default(),
        });
        answer_in(joins, participant, self.session.clone(), shutdown).await
    }
}

/// A driver's connection to a served model: its participant in the
/// model's session (or the session it invited the model into), and the
/// link.
pub struct ModelConnection {
    /// The link to the model.
    pub link: ModelLink,
    model: ParticipantId,
    participant: Participant,
    lockstep: Option<zenoh::Session>,
    /// The model's own session, when this connects there.
    own: Option<SessionAddress>,
    /// The session the model was invited into, which it leaves at close.
    invited: Option<String>,
}

impl ModelConnection {
    /// Connects to the model at the participant address `address`: joins
    /// its session as `id` (never hosting it: the model's process does)
    /// and waits up to `timeout` for the model to answer.
    pub async fn connect(address: &str, id: &str, timeout: Duration) -> Result<Self, GymError> {
        let parsed = Address::participant(address).map_err(|e| GymError::Config(e.to_string()))?;
        let model = ParticipantId::new(parsed.participant.clone().unwrap_or_default())
            .map_err(|e| GymError::Config(e.to_string()))?;
        let own = parsed.joining()?;
        let mut connection = Self::join(&own, &model, id, timeout).await?;
        connection.own = Some(own);
        Ok(connection)
    }

    /// Invites the model this connects to into `session` (a session
    /// address, which this driver runs in) and connects to it there,
    /// leaving the model's own session: predicts then stay in `session`.
    /// Refused when the model can't reach it.
    pub async fn invite(
        self,
        session: &str,
        id: &str,
        timeout: Duration,
    ) -> Result<Self, GymError> {
        let model = self.model.clone();
        let request = ModelRequest::Join {
            session: session.to_string(),
        };
        let invited = self.link.call(&request, Some(timeout)).await;
        self.close().await;
        match invited? {
            (ModelReply::Done, _) => {}
            (ModelReply::Fault(fault), _) => return Err(GymError::Config(fault.message)),
            (other, _) => {
                return Err(GymError::Malformed(format!(
                    "a model answered an invitation with {other:?}"
                )));
            }
        }
        let address: SessionAddress = session.parse()?;
        let mut connection = Self::join(&address, &model, id, timeout).await?;
        connection.invited = Some(session.to_string());
        Ok(connection)
    }

    /// Joins `session` as `id` and waits up to `timeout` for `model` to
    /// answer there.
    async fn join(
        session: &SessionAddress,
        model: &ParticipantId,
        id: &str,
        timeout: Duration,
    ) -> Result<Self, GymError> {
        let participant = session
            .join(id, vec![Role::Controller], PriorityClass::Model, timeout)
            .await?;
        let lockstep_session = lockstep_session(participant.transport(), session).await?;
        let link = ModelLink::new(lockstep_session.clone(), participant.keys().model(model));
        link.wait_ready(timeout).await.map_err(|err| match err {
            GymError::Timeout(_) => GymError::Session(SessionError::Config(format!(
                "no model {model} answered at {session}/{model} within {timeout:?}"
            ))),
            other => other,
        })?;
        let lockstep = crate::lockstep::session::owns_links(participant.transport())
            .then_some(lockstep_session);
        Ok(Self {
            link,
            model: model.clone(),
            participant,
            lockstep,
            own: None,
            invited: None,
        })
    }

    /// Whether the model answers in `session` (a session address): its own
    /// session, which it needs no invitation into.
    pub fn answers_in(&self, session: &str) -> bool {
        let Some(own) = &self.own else {
            return false;
        };
        session
            .parse::<SessionAddress>()
            .is_ok_and(|session| session.base == own.base && session.session == own.session)
    }

    /// The workflow editions the model declared in its session, from the
    /// roster; `None` when it declared none.
    pub fn workflow_editions(&self) -> Option<rlmesh_proto::SessionOffer> {
        let snapshot = self.participant.watch().borrow().clone()?;
        let model = snapshot.roster.into_iter().find(|p| p.id == self.model)?;
        (!model.workflow_editions.is_empty()).then(|| rlmesh_proto::SessionOffer {
            editions: model.workflow_editions,
            preferred: Some(model.workflow_edition).filter(|edition| !edition.is_empty()),
        })
    }

    /// Leaves the model's session (asking an invited model to leave it).
    pub async fn close(self) {
        if let Some(session) = &self.invited {
            let leave = ModelRequest::Leave {
                session: session.clone(),
            };
            if let Err(err) = self.link.call(&leave, Some(Duration::from_secs(5))).await {
                tracing::debug!(%err, "model client: leave");
            }
        }
        if let Err(err) = self.participant.detach().await {
            tracing::debug!(%err, "model client: detach");
        }
        if let Some(lockstep) = self.lockstep {
            let _ = lockstep.close().await;
        }
    }
}
