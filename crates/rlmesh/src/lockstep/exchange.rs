//! The lockstep exchange (§12.1-§12.3): one request, one reply, per step.
//!
//! ```text
//! driver ──LockstepStep{epoch, step, lanes, commands}──▶ env
//! driver ◀──LockstepReply{step, observations, reward, terminated, truncated, episode_ids}── env
//! ```
//!
//! The env side is a [`LockstepHandler`] behind a `Gate`: before every
//! step it checks the step's epoch and lease token against its own
//! authority view (two atomic loads) and the e-stop latch, with no host
//! round trip. The driver side reaches it through a [`LockstepLink`]:
//!
//! | Placement | Link |
//! |---|---|
//! | same process | [`DirectLink`]: a function call, no serialisation |
//! | same host or remote | [`ZenohLink`]: a query on `l/{robot}` over the lockstep class session, answered by [`serve_zenoh`] |
//!
//! Over Zenoh every request is a `LockstepRequest` sealed under the lease's
//! `k_cmd` (design §8.6): a step, a reset, a render (a lane's frame) or a
//! probe, with a request id. The env answers a `LockstepResponse` with that
//! id, sealed under the lease's `k_obs`: the observation, the frame's PNG
//! bytes (empty for none), the probe, or the env's failure as JSON
//! (`{code, message, recoverable}`). A request that doesn't open under a
//! lease key the env holds gets a plain error reply, which the driver takes
//! only as a hint. Both ends check the request's lease token against the
//! key's.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_session::Sealer;
use rlmesh_session::transport::payload_bytes;
use rlmesh_session::wire::timing::ns;
use rlmesh_session::wire::{
    AuthoritySnapshot, ChannelId, LeaseToken, LockstepOp, LockstepReply, LockstepRequest,
    LockstepReset, LockstepResponse, LockstepResult, LockstepStep, Mode, ParticipantId,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use zenoh::qos::{CongestionControl, Priority};
use zenoh::query::ConsolidationMode;

use crate::lockstep::error::{GymError, failure_from_json, failure_to_json};

/// One lockstep request.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Reset lanes.
    Reset(LockstepReset),
    /// Step lanes: exactly one wrapped step.
    Step(LockstepStep),
}

impl Request {
    /// The epoch it was issued in.
    pub fn epoch(&self) -> rlmesh_session::wire::Epoch {
        match self {
            Self::Reset(reset) => reset.epoch,
            Self::Step(step) => step.epoch,
        }
    }

    /// The driver's lease token: a reset needs the lease, like a step.
    pub fn lease_token(&self) -> LeaseToken {
        match self {
            Self::Reset(reset) => reset.lease_token,
            Self::Step(step) => step.lease_token,
        }
    }
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

    /// Waits until everything answered so far is recorded; by default
    /// nothing is.
    async fn flush(&self) {}
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
        let (asked, token) = (request.epoch().0, request.lease_token().0);
        if asked != epoch {
            return Err(GymError::Refused(format!(
                "epoch {asked} is stale; the session is in epoch {epoch}"
            )));
        }
        let held = self.token.load(Ordering::Acquire);
        if held == 0 || token != held {
            return Err(GymError::Refused(format!(
                "lease token {token} does not hold the robot's commandable parts"
            )));
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
        let arrived = std::time::Instant::now();
        let waited = self.gate.check(&request).is_err();
        if waited {
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
        let queue_ns = waited.then(|| ns(arrived.elapsed())).flatten();
        let mut reply = self.inner.handle(request).await?;
        if queue_ns.is_some() {
            reply.timing.get_or_insert_default().queue_ns = queue_ns;
        }
        Ok(reply)
    }

    async fn render(&self, lane: u32) -> Result<Option<Bytes>, GymError> {
        self.inner.render(lane).await
    }

    fn concurrent(&self) -> bool {
        self.inner.concurrent()
    }

    async fn flush(&self) {
        self.inner.flush().await;
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

/// The same-process link: a function call, no serialisation. It holds
/// the handler weakly, so a client outliving the env's server fails as a
/// remote one would instead of stepping an env that stopped serving.
pub struct DirectLink(pub Weak<dyn LockstepHandler>);

impl DirectLink {
    fn handler(&self) -> Result<Arc<dyn LockstepHandler>, GymError> {
        self.0
            .upgrade()
            .ok_or_else(|| GymError::Closed("the env stopped serving".into()))
    }
}

#[async_trait]
impl LockstepLink for DirectLink {
    async fn call(
        &self,
        request: Request,
        timeout: Option<Duration>,
    ) -> Result<LockstepReply, GymError> {
        within(timeout, self.handler()?.handle(request)).await
    }

    async fn render(
        &self,
        lane: u32,
        timeout: Option<Duration>,
    ) -> Result<Option<Bytes>, GymError> {
        within(timeout, self.handler()?.render(lane)).await
    }
}

/// How long a lockstep query waits when the request sets no deadline.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);

/// A query on `l/{robot}` over the lockstep class session, sealed under
/// the lease's keys.
pub struct ZenohLink {
    session: zenoh::Session,
    key: String,
    sealer: Sealer,
    robot: ParticipantId,
    token: LeaseToken,
    last_request: AtomicU64,
}

impl ZenohLink {
    /// A link to `robot`'s queryable on `key`, under the lease `token`
    /// whose keys `sealer` holds.
    pub fn new(
        session: zenoh::Session,
        key: String,
        sealer: Sealer,
        robot: ParticipantId,
        token: LeaseToken,
    ) -> Self {
        Self {
            session,
            key,
            sealer,
            robot,
            token,
            last_request: AtomicU64::new(0),
        }
    }

    /// Waits until the env's queryable answers a probe (its declaration
    /// has reached this session), probing every 20 ms for up to `timeout`.
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), GymError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            match self
                .exchange(LockstepOp::Probe, Some(Duration::from_millis(500)))
                .await
            {
                Ok(_) => return Ok(()),
                Err(_) if tokio::time::Instant::now() < deadline => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(_) => return Err(GymError::Timeout(timeout)),
            }
        }
    }

    /// One sealed request: the env's sealed result. An unsealed error reply
    /// is only a hint (anyone can answer `l/{robot}`), returned when no
    /// sealed reply came.
    async fn exchange(
        &self,
        op: LockstepOp,
        timeout: Option<Duration>,
    ) -> Result<LockstepResult, GymError> {
        let timeout = timeout.unwrap_or(DEFAULT_TIMEOUT);
        let request_id = self.last_request.fetch_add(1, Ordering::Relaxed) + 1;
        let sealed = self.sealer.seal_lockstep_request(
            &self.key,
            &self.robot,
            self.token,
            &LockstepRequest { request_id, op },
        )?;
        let replies = self
            .session
            .get(&self.key)
            .payload(sealed)
            .consolidation(ConsolidationMode::None)
            .priority(Priority::RealTime)
            .congestion_control(CongestionControl::Block)
            .express(true)
            .timeout(timeout)
            .await
            .map_err(|e| GymError::Closed(e.to_string()))?;
        let mut hint = None;
        while let Ok(reply) = replies.recv_async().await {
            match reply.result() {
                Ok(sample) => {
                    let msg = sample.payload().to_bytes().into_owned();
                    match self.sealer.open_lockstep_response(
                        &self.key,
                        msg,
                        &self.robot,
                        self.token,
                    ) {
                        Ok(response) if response.request_id == request_id => {
                            return match response.result {
                                LockstepResult::Failure(json) => {
                                    Err(failure(json.as_bytes())
                                        .unwrap_or(GymError::Timeout(timeout)))
                                }
                                result => Ok(result),
                            };
                        }
                        Ok(_) => {}
                        Err(err) => debug!(%err, "lockstep reply dropped"),
                    }
                }
                Err(err) => hint = failure(&payload_bytes(err.payload())),
            }
        }
        // No sealed reply: the env's plain refusal, or a Zenoh-level error
        // (the queryable is gone).
        Err(hint.unwrap_or(GymError::Timeout(timeout)))
    }
}

/// The env's failure in `json`, if it names one.
fn failure(json: &[u8]) -> Option<GymError> {
    failure_from_json(json)
        .filter(|f| !f.message.is_empty())
        .map(GymError::Env)
}

#[async_trait]
impl LockstepLink for ZenohLink {
    async fn call(
        &self,
        request: Request,
        timeout: Option<Duration>,
    ) -> Result<LockstepReply, GymError> {
        let op = match request {
            Request::Reset(reset) => LockstepOp::Reset(reset),
            Request::Step(step) => LockstepOp::Step(step),
        };
        match self.exchange(op, timeout).await? {
            LockstepResult::Obs(reply) => Ok(*reply),
            other => Err(GymError::Malformed(format!("{other:?} answers a step"))),
        }
    }

    async fn render(
        &self,
        lane: u32,
        timeout: Option<Duration>,
    ) -> Result<Option<Bytes>, GymError> {
        match self.exchange(LockstepOp::Render(lane), timeout).await? {
            LockstepResult::Frame(frame) => Ok((!frame.is_empty()).then_some(frame)),
            other => Err(GymError::Malformed(format!("{other:?} answers a render"))),
        }
    }
}

/// Answers request `request_id` under the lease `token`'s `k_obs`.
async fn answer(
    query: &zenoh::query::Query,
    key: &str,
    sealer: &Sealer,
    token: LeaseToken,
    request_id: u64,
    result: Result<LockstepResult, GymError>,
) {
    let result = result.unwrap_or_else(|err| {
        LockstepResult::Failure(
            String::from_utf8_lossy(&failure_to_json(&err.to_failure())).into_owned(),
        )
    });
    let sealed =
        sealer.seal_lockstep_response(key, token, &LockstepResponse { request_id, result });
    let sent = match sealed {
        Ok(sealed) => query.reply(key, sealed).express(true).await,
        Err(err) => {
            warn!(%err, "lockstep reply not sealed");
            return;
        }
    };
    if let Err(err) = sent {
        warn!(%err, "lockstep reply not sent");
    }
}

/// `handled`'s reply with the exchange's own timing: the wire decode added
/// to the handler's, the handler's whole call as its work when it measured
/// none, and the total since arrival.
async fn timed(
    handled: impl std::future::Future<Output = Result<LockstepReply, GymError>>,
    arrived: std::time::Instant,
    decode: std::time::Duration,
) -> Result<LockstepReply, GymError> {
    let called = std::time::Instant::now();
    let mut reply = handled.await?;
    let call = called.elapsed();
    let timing = reply.timing.get_or_insert_default();
    timing.decode_ns = ns(decode).map(|wire| wire + timing.decode_ns.unwrap_or(0));
    if timing.user_ns.is_none() {
        timing.user_ns = ns(call);
    }
    timing.total_ns = ns(arrived.elapsed());
    Ok(reply)
}

/// Serves `handler` on `key` until `cancel` fires, opening requests and
/// sealing replies under the lease keys `sealer` holds.
pub async fn serve_zenoh(
    session: zenoh::Session,
    key: String,
    handler: Arc<dyn LockstepHandler>,
    cancel: CancellationToken,
    sealer: Sealer,
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
        let arrived = std::time::Instant::now();
        let msg = query
            .payload()
            .map(|p| p.to_bytes().into_owned())
            .unwrap_or_default();
        let (request, token) = match sealer.open_lockstep_request(&key, msg) {
            Ok(opened) => opened,
            Err(err) => {
                // Not a lease holder's: a plain refusal, a hint only.
                let refused = GymError::Refused(err.to_string());
                if let Err(err) = query
                    .reply_err(failure_to_json(&refused.to_failure()))
                    .await
                {
                    debug!(%err, "lockstep refusal not sent");
                }
                continue;
            }
        };
        let request_id = request.request_id;
        let request = match request.op {
            LockstepOp::Probe => {
                answer(
                    &query,
                    &key,
                    &sealer,
                    token,
                    request_id,
                    Ok(LockstepResult::Probe),
                )
                .await;
                continue;
            }
            LockstepOp::Render(lane) => {
                let frame = handler
                    .render(lane)
                    .await
                    .map(|frame| LockstepResult::Frame(frame.unwrap_or_default()));
                answer(&query, &key, &sealer, token, request_id, frame).await;
                continue;
            }
            LockstepOp::Reset(reset) => Request::Reset(reset),
            LockstepOp::Step(step) => Request::Step(step),
        };
        // The request's lease is the one whose key it came under.
        if request.lease_token() != token {
            let refused = GymError::Refused(format!(
                "lease token {} under the key of {token}",
                request.lease_token()
            ));
            answer(&query, &key, &sealer, token, request_id, Err(refused)).await;
            continue;
        }
        let decode_ns = arrived.elapsed();
        // Lane-scoped requests of a lane endpoint run side by side (as on
        // v1's Join stream); everything else is served in order, inline.
        let lane_scoped = match &request {
            Request::Reset(reset) => !reset.lanes.is_empty(),
            Request::Step(step) => !step.lanes.is_empty(),
        };
        if concurrent && lane_scoped {
            let (handler, key, sealer) = (handler.clone(), key.clone(), sealer.clone());
            tokio::spawn(async move {
                let result = timed(handler.handle(request), arrived, decode_ns).await;
                answer(
                    &query,
                    &key,
                    &sealer,
                    token,
                    request_id,
                    result.map(|reply| LockstepResult::Obs(Box::new(reply))),
                )
                .await;
            });
        } else {
            let result = timed(handler.handle(request), arrived, decode_ns).await;
            answer(
                &query,
                &key,
                &sealer,
                token,
                request_id,
                result.map(|reply| LockstepResult::Obs(Box::new(reply))),
            )
            .await;
        }
    }
    drop(queryable);
    Ok(())
}
