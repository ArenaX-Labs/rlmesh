//! The lockstep exchange (§12.1-§12.3): one request, one reply, per step.
//!
//! ```text
//! driver ──LockstepStep{epoch, step, lanes, commands}──▶ env
//! driver ◀──LockstepReply{step, observations, reward, terminated, truncated, episode_ids}── env
//! ```
//!
//! The env side is a [`LockstepHandler`] behind a [`Gate`]: before every
//! step it checks the step's epoch and lease token against its own
//! authority view (two atomic loads) and the e-stop latch, with no host
//! round trip. The driver side reaches it through a [`LockstepLink`]:
//!
//! | Placement | Link |
//! |---|---|
//! | same process | [`DirectLink`]: a function call, no serialisation |
//! | same host or remote | [`ZenohLink`]: a query on `l/{robot}` over the lockstep class session, answered by [`serve_zenoh`] |
//!
//! A reset is the same query with the selector parameter `reset`, a render
//! the selector parameter `render` (the lane as a little-endian `u32` in, the
//! frame's PNG bytes out, empty for none); an error reply carries the env's
//! failure as JSON (`{code, message, recoverable}`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_session::transport::payload_bytes;
use rlmesh_session::wire::codec;
use rlmesh_session::wire::{
    AuthoritySnapshot, ChannelId, LockstepReply, LockstepReset, LockstepStep, Mode, ParticipantId,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use zenoh::qos::{CongestionControl, Priority};
use zenoh::query::ConsolidationMode;

use crate::lockstep::error::{GymError, failure_from_json, failure_to_json};

/// The selector parameter of a reset query.
const RESET: &str = "reset";
/// The selector parameter of a render query.
const RENDER: &str = "render";
/// The selector parameter of a probe: answered with an empty reply, so a
/// driver knows the route to the env is up before its first request.
const PROBE: &str = "probe";

/// One lockstep request.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Reset lanes.
    Reset(LockstepReset),
    /// Step lanes: exactly one wrapped step.
    Step(LockstepStep),
}

/// The env side of the exchange.
#[async_trait]
pub trait LockstepHandler: Send + Sync + 'static {
    /// Answers one request.
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError>;

    /// Renders lane `lane`'s frame as PNG bytes; by default there is none.
    async fn render(&self, _lane: u32) -> Result<Option<Bytes>, GymError> {
        Ok(None)
    }

    /// Whether several lane-scoped requests may run at once (a lane
    /// endpoint, v1's `subset_step`).
    fn concurrent(&self) -> bool {
        false
    }
}

/// The env's local authority view, read per step.
#[derive(Debug)]
pub struct Gate {
    enforce: bool,
    epoch: AtomicU64,
    token: AtomicU64,
    estop: AtomicBool,
    changed: tokio::sync::Notify,
}

/// How long a refused request waits for the env's view to catch up with
/// the driver's (a lease the driver holds before the env has seen it).
const VIEW_CATCH_UP: Duration = Duration::from_secs(1);

impl Gate {
    /// A gate that checks nothing (no session: tests, in-process pairs).
    pub fn open() -> Arc<Self> {
        Arc::new(Self {
            enforce: false,
            epoch: AtomicU64::new(0),
            token: AtomicU64::new(0),
            estop: AtomicBool::new(false),
            changed: tokio::sync::Notify::new(),
        })
    }

    /// A gate that refuses everything until [`Self::observe`] sees a lease.
    pub fn closed() -> Arc<Self> {
        Arc::new(Self {
            enforce: true,
            epoch: AtomicU64::new(0),
            token: AtomicU64::new(0),
            estop: AtomicBool::new(false),
            changed: tokio::sync::Notify::new(),
        })
    }

    /// Takes the epoch, the lease on the robot's commandable parts and the
    /// e-stop latch from an authority snapshot.
    pub fn observe(
        &self,
        snapshot: &AuthoritySnapshot,
        robot: &ParticipantId,
        commands: &[ChannelId],
    ) {
        let token = commands
            .first()
            .and_then(|c| snapshot.leases.iter().find(|l| l.channel == *c))
            .filter(|lease| {
                commands.iter().all(|c| {
                    snapshot
                        .leases
                        .iter()
                        .any(|l| l.channel == *c && l.token == lease.token)
                })
            })
            .map_or(0, |lease| lease.token.0);
        let estop = snapshot.estop_latched || snapshot.mode == Mode::Estop;
        // Publish the epoch last: a step that reads the new epoch also sees
        // its token.
        self.token.store(token, Ordering::Release);
        self.estop.store(estop, Ordering::Release);
        self.epoch.store(snapshot.epoch.0, Ordering::Release);
        self.changed.notify_waiters();
        debug!(%robot, epoch = snapshot.epoch.0, token, estop, "lockstep gate");
    }

    /// Checks a request against the view: the e-stop between steps, the
    /// epoch, and a step's lease token.
    pub fn check(&self, request: &Request) -> Result<(), GymError> {
        if !self.enforce {
            return Ok(());
        }
        if self.estop.load(Ordering::Acquire) {
            return Err(GymError::Refused("the session is e-stopped".into()));
        }
        let epoch = self.epoch.load(Ordering::Acquire);
        let (asked, token) = match request {
            Request::Reset(reset) => (reset.epoch.0, None),
            Request::Step(step) => (step.epoch.0, Some(step.lease_token.0)),
        };
        if asked != epoch {
            return Err(GymError::Refused(format!(
                "epoch {asked} is stale; the session is in epoch {epoch}"
            )));
        }
        if let Some(token) = token {
            let held = self.token.load(Ordering::Acquire);
            if held == 0 || token != held {
                return Err(GymError::Refused(format!(
                    "lease token {token} does not hold the robot's commandable parts"
                )));
            }
        }
        Ok(())
    }
}

/// A handler behind its gate.
pub struct Gated {
    gate: Arc<Gate>,
    inner: Arc<dyn LockstepHandler>,
}

impl Gated {
    /// `inner` behind `gate`.
    pub fn new(gate: Arc<Gate>, inner: Arc<dyn LockstepHandler>) -> Arc<Self> {
        Arc::new(Self { gate, inner })
    }
}

#[async_trait]
impl LockstepHandler for Gated {
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError> {
        if self.gate.check(&request).is_err() {
            // The driver may hold a lease (or epoch) the env's view has not
            // seen yet: wait for the view, then decide.
            let deadline = tokio::time::Instant::now() + VIEW_CATCH_UP;
            loop {
                let changed = self.gate.changed.notified();
                match self.gate.check(&request) {
                    Ok(()) => break,
                    Err(refused) if tokio::time::Instant::now() >= deadline => {
                        return Err(refused);
                    }
                    Err(_) => {}
                }
                let _ = tokio::time::timeout_at(deadline, changed).await;
            }
        }
        self.inner.handle(request).await
    }

    async fn render(&self, lane: u32) -> Result<Option<Bytes>, GymError> {
        self.inner.render(lane).await
    }

    fn concurrent(&self) -> bool {
        self.inner.concurrent()
    }
}

/// The driver side's way to the env.
#[async_trait]
pub trait LockstepLink: Send + Sync + 'static {
    /// Sends one request and waits for its reply.
    async fn call(
        &self,
        request: Request,
        timeout: Option<Duration>,
    ) -> Result<LockstepReply, GymError>;

    /// Asks the env for lane `lane`'s frame (PNG bytes).
    async fn render(&self, lane: u32, timeout: Option<Duration>)
    -> Result<Option<Bytes>, GymError>;
}

/// `future` bounded by `timeout`, when there is one.
async fn within<T>(
    timeout: Option<Duration>,
    future: impl std::future::Future<Output = Result<T, GymError>>,
) -> Result<T, GymError> {
    match timeout {
        Some(t) => tokio::time::timeout(t, future)
            .await
            .map_err(|_| GymError::Timeout(t))?,
        None => future.await,
    }
}

/// The same-process link: a function call, no serialisation.
pub struct DirectLink(pub Arc<dyn LockstepHandler>);

#[async_trait]
impl LockstepLink for DirectLink {
    async fn call(
        &self,
        request: Request,
        timeout: Option<Duration>,
    ) -> Result<LockstepReply, GymError> {
        within(timeout, self.0.handle(request)).await
    }

    async fn render(
        &self,
        lane: u32,
        timeout: Option<Duration>,
    ) -> Result<Option<Bytes>, GymError> {
        within(timeout, self.0.render(lane)).await
    }
}

/// How long a lockstep query waits when the request sets no deadline.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// A query on `l/{robot}` over the lockstep class session.
pub struct ZenohLink {
    session: zenoh::Session,
    key: String,
    reset_selector: String,
    render_selector: String,
}

impl ZenohLink {
    /// A link to the queryable on `key`.
    pub fn new(session: zenoh::Session, key: String) -> Self {
        Self {
            reset_selector: format!("{key}?{RESET}"),
            render_selector: format!("{key}?{RENDER}"),
            session,
            key,
        }
    }

    /// Waits until the env's queryable answers (its declaration has
    /// reached this session), probing every 20 ms for up to `timeout`.
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
}

/// Encodes a request: its selector parameters and its payload.
fn encode(request: &Request) -> (bool, Vec<u8>) {
    match request {
        Request::Reset(reset) => (true, codec::encode_lockstep_reset(reset)),
        Request::Step(step) => (false, codec::encode_lockstep_step(step)),
    }
}

impl ZenohLink {
    /// One query on `selector`: the reply's payload, or the env's error.
    async fn query(
        &self,
        selector: &str,
        payload: Vec<u8>,
        timeout: Option<Duration>,
    ) -> Result<Bytes, GymError> {
        let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
        let replies = self
            .session
            .get(selector)
            .payload(payload)
            .consolidation(ConsolidationMode::None)
            .priority(Priority::RealTime)
            .congestion_control(CongestionControl::Block)
            .express(true)
            .timeout(timeout)
            .await
            .map_err(|e| GymError::Closed(e.to_string()))?;
        let reply = replies
            .recv_async()
            .await
            .map_err(|_| GymError::Timeout(timeout))?;
        match reply.result() {
            Ok(sample) => Ok(payload_bytes(sample.payload())),
            Err(err) => match failure_from_json(&payload_bytes(err.payload())) {
                Some(failure) if !failure.message.is_empty() => Err(GymError::Env(failure)),
                // A Zenoh-level error (the queryable is gone).
                _ => Err(GymError::Timeout(timeout)),
            },
        }
    }
}

#[async_trait]
impl LockstepLink for ZenohLink {
    async fn call(
        &self,
        request: Request,
        timeout: Option<Duration>,
    ) -> Result<LockstepReply, GymError> {
        let (reset, payload) = encode(&request);
        let selector = if reset {
            self.reset_selector.as_str()
        } else {
            self.key.as_str()
        };
        let reply = self.query(selector, payload, timeout).await?;
        Ok(codec::decode_lockstep_reply(reply)?)
    }

    async fn render(
        &self,
        lane: u32,
        timeout: Option<Duration>,
    ) -> Result<Option<Bytes>, GymError> {
        let frame = self
            .query(&self.render_selector, lane.to_le_bytes().to_vec(), timeout)
            .await?;
        Ok((!frame.is_empty()).then_some(frame))
    }
}

fn decode_request(query: &zenoh::query::Query) -> Result<Request, GymError> {
    let payload = query.payload().map(payload_bytes).unwrap_or_default();
    if query.parameters().contains_key(RESET) {
        Ok(Request::Reset(codec::decode_lockstep_reset(&payload)?))
    } else {
        Ok(Request::Step(codec::decode_lockstep_step(payload)?))
    }
}

async fn answer(query: &zenoh::query::Query, key: &str, result: Result<LockstepReply, GymError>) {
    answer_bytes(
        query,
        key,
        result.map(|reply| codec::encode_lockstep_reply(&reply)),
    )
    .await;
}

async fn answer_bytes(query: &zenoh::query::Query, key: &str, result: Result<Vec<u8>, GymError>) {
    let sent = match result {
        Ok(payload) => query.reply(key, payload).express(true).await,
        Err(err) => query.reply_err(failure_to_json(&err.to_failure())).await,
    };
    if let Err(err) = sent {
        warn!(%err, "lockstep reply not sent");
    }
}

/// Serves `handler` on `key` until `cancel` fires.
pub async fn serve_zenoh(
    session: zenoh::Session,
    key: String,
    handler: Arc<dyn LockstepHandler>,
    cancel: CancellationToken,
) -> Result<(), GymError> {
    let queryable = session
        .declare_queryable(&key)
        .complete(true)
        .await
        .map_err(|e| GymError::Closed(e.to_string()))?;
    let concurrent = handler.concurrent();
    loop {
        let query = tokio::select! {
            () = cancel.cancelled() => break,
            query = queryable.recv_async() => match query {
                Ok(query) => query,
                Err(_) => break,
            },
        };
        if query.parameters().contains_key(PROBE) {
            answer(&query, &key, Ok(LockstepReply::default())).await;
            continue;
        }
        if query.parameters().contains_key(RENDER) {
            let payload = query.payload().map(payload_bytes).unwrap_or_default();
            let result = match <[u8; 4]>::try_from(payload.as_ref()) {
                Ok(lane) => handler
                    .render(u32::from_le_bytes(lane))
                    .await
                    .map(|frame| frame.map(|f| f.to_vec()).unwrap_or_default()),
                Err(_) => Err(GymError::Malformed("a render names one u32 lane".into())),
            };
            answer_bytes(&query, &key, result).await;
            continue;
        }
        let request = match decode_request(&query) {
            Ok(request) => request,
            Err(err) => {
                answer(&query, &key, Err(err)).await;
                continue;
            }
        };
        // Lane-scoped requests of a lane endpoint run side by side (as on
        // v1's Join stream); everything else is served in order, inline.
        let lane_scoped = match &request {
            Request::Reset(reset) => !reset.lanes.is_empty(),
            Request::Step(step) => !step.lanes.is_empty(),
        };
        if concurrent && lane_scoped {
            let (handler, key) = (handler.clone(), key.clone());
            tokio::spawn(async move {
                let result = handler.handle(request).await;
                answer(&query, &key, result).await;
            });
        } else {
            let result = handler.handle(request).await;
            answer(&query, &key, result).await;
        }
    }
    drop(queryable);
    Ok(())
}
