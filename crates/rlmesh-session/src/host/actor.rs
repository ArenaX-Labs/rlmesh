//! The host task: its loop, the per-verb bookkeeping types, and how
//! authority events are recorded. Each concern's methods are in a submodule.

mod access;
mod ctl;
mod datagram;

pub(super) use datagram::DatagramSession;
mod grant;
mod handshake;
mod health;
mod keys;
#[cfg(feature = "fleet")]
mod lobby;
mod publish;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::wire::{
    ChunkTiming, ClockRefusal, CtlError, DatagramStats, Epoch, IssuedKey, Lease, LeaseToken,
    ParticipantId,
};
use rlmesh_session_core::auth::issuer::GrantId;
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zenoh::query::Query;

use crate::authority::{AuthorityEvent, DetachReason};
use crate::clock::OffsetEstimator;
use crate::edge::core::Rejection;
use crate::host::peer::Peer;
use crate::host::{AccessChange, Change, HostActor};
#[cfg(feature = "fleet")]
use crate::lobby::LobbyResult;
use crate::metrics::LinkStats;

/// How often the host publishes `host/health`.
const HEALTH_PERIOD: Duration = Duration::from_secs(1);

/// A participant's timing link, as the host measures it.
pub(super) struct Link {
    /// The generation it measures: a new process starts a new link.
    generation: u32,
    estimator: OffsetEstimator,
    stats: LinkStats,
    /// Edges: chunks refused since attach, by reason (its latest heartbeat).
    rejected: Vec<(Rejection, u64)>,
    /// Edges: why the clocks are refusing chunks (its latest heartbeat).
    clock_refused: Vec<ClockRefusal>,
    /// Edges: how each commanded part's chunks arrived, and how many were
    /// accepted (its latest heartbeat).
    chunk_timing: Vec<ChunkTiming>,
    chunks_accepted: u64,
    /// A board on the datagram path: its end's counters (its latest
    /// heartbeat).
    datagram: Option<DatagramStats>,
}

type Fifo<T> = zenoh::handlers::FifoChannelHandler<T>;
type Queries = zenoh::query::Queryable<Fifo<Query>>;

/// A lobby verb: its sender, the sender's generation and the request id.
#[cfg(feature = "fleet")]
pub(super) type LobbyKey = (ParticipantId, u32, u64);

/// A waiting lobby verb's answer, back at the host task.
#[cfg(feature = "fleet")]
pub(super) type LobbyDone = (LobbyKey, LobbyResult);

/// A lobby verb waiting for its answer, and the latest query asking for it
/// (a retry replaces it: the earlier one's caller has timed out).
#[cfg(feature = "fleet")]
pub(super) struct LobbyWaiting {
    peer: Arc<Peer>,
    query: Option<Query>,
    task: tokio::task::AbortHandle,
}

/// A grant (or rebind) whose keys are on their way to its robots: the verb,
/// what it was prepared against, and every query waiting for its answer.
pub(super) struct Waiting {
    peer: Arc<Peer>,
    request: crate::wire::Request,
    token: LeaseToken,
    epoch: Epoch,
    robots: Vec<(ParticipantId, u32)>,
    queries: Vec<Query>,
}

/// What the host does with a verb.
enum Outcome {
    /// Answer it now.
    Reply(Result<(), CtlError>, Vec<Lease>, Vec<IssuedKey>),
    /// A candidate sent something other than `Attach`: dropped unanswered.
    Drop,
    /// A grant waiting for its robots: answered once they acknowledged.
    Deferred(GrantId),
    /// A lobby verb, answered now.
    #[cfg(feature = "fleet")]
    Lobby(LobbyResult),
    /// A lobby verb answered later (a long poll).
    #[cfg(feature = "fleet")]
    Later(tokio::sync::oneshot::Receiver<LobbyResult>),
}

impl Outcome {
    fn refused(err: CtlError) -> Self {
        Self::Reply(Err(err), Vec::new(), Vec::new())
    }
}

/// Whether the expiry tick may judge silence. A host that did not run for a
/// while (its process was descheduled or frozen) was not listening in that
/// time: heartbeats sent meanwhile are still on their way to it, and
/// participants in its own process stalled with it. Judging on the first
/// tick back would lapse the leases of healthy holders, so that tick starts
/// one lease grace of listening instead. The judgement after it always
/// runs, so a host that keeps stalling still expires what is silent.
struct Heard {
    last: tokio::time::Instant,
    listening_until: Option<tokio::time::Instant>,
    stall: Duration,
    grace: Duration,
}

impl Heard {
    fn new(now: tokio::time::Instant, tick: Duration, grace: Duration) -> Self {
        Self {
            last: now,
            listening_until: None,
            // Well past a late tick, well short of a lapse.
            stall: (grace / 2).max(tick * 4),
            grace,
        }
    }

    /// Whether the host is listening after a stall.
    fn listening(&self) -> bool {
        self.listening_until.is_some()
    }

    /// Whether the tick at `now` judges leases and presence.
    fn judge(&mut self, now: tokio::time::Instant) -> bool {
        let gap = now.saturating_duration_since(self.last);
        self.last = now;
        match self.listening_until {
            Some(until) if now < until => false,
            Some(_) => {
                self.listening_until = None;
                true
            }
            None if gap > self.stall => {
                debug!(?gap, "host stalled: listening before judging leases");
                self.listening_until = Some(now + self.grace);
                false
            }
            None => true,
        }
    }
}

impl HostActor {
    pub(super) async fn run(
        &mut self,
        cancel: CancellationToken,
        mut heartbeats: mpsc::Receiver<(ParticipantId, crate::wire::Heartbeat, i64)>,
        mut access: mpsc::UnboundedReceiver<AccessChange>,
        (handshakes, ctl, state, views): (Queries, Queries, Queries, Queries),
        presence: zenoh::pubsub::Subscriber<Fifo<zenoh::sample::Sample>>,
    ) {
        let interval = |period: Duration| {
            let mut interval = tokio::time::interval(period.max(Duration::from_millis(1)));
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            interval
        };
        let mut tick = interval(self.config.tick);
        let mut heard = Heard::new(
            tokio::time::Instant::now(),
            self.config.tick,
            self.config.lease_grace,
        );
        let mut republish = interval(self.config.republish);
        let health_log = self.config.health_log;
        let mut health = interval(health_log.unwrap_or(Duration::from_secs(1)));
        let mut health_out = interval(HEALTH_PERIOD);

        loop {
            tokio::select! {
                () = cancel.cancelled() => break,
                query = handshakes.recv_async() => match query {
                    Ok(query) => self.on_handshake(query).await,
                    Err(_) => break,
                },
                query = ctl.recv_async() => match query {
                    Ok(query) => {
                        self.on_ctl(query).await;
                        self.sync_history();
                        self.sync_streams();
                    }
                    Err(_) => break,
                },
                query = state.recv_async() => match query {
                    Ok(query) => self.on_state_query(query).await,
                    Err(_) => break,
                },
                query = views.recv_async() => match query {
                    Ok(query) => self.on_view_query(query).await,
                    Err(_) => break,
                },
                Some((id, heartbeat, t2)) = heartbeats.recv() => {
                    self.on_heartbeat(id, heartbeat, t2);
                }
                Some(done) = self.lobby_answer() => self.finish_lobby(done).await,
                Some((change, done)) = access.recv() => {
                    let applied = match change {
                        Change::Access(device, entry) => self.set_access(&device, entry).await,
                        Change::Admit(device) => self.admit(device).await,
                        Change::Revoke(device) => self.revoke(&device).await,
                        Change::Pin(robot, pin) => {
                            self.pin(&robot, pin).await;
                            Ok(())
                        }
                        #[cfg(feature = "fleet")]
                        Change::Close(code, by, detail) => {
                            self.close_scene(code, &by, &detail).await
                        }
                    };
                    let _ = done.send(applied);
                }
                sample = presence.recv_async() => match sample {
                    Ok(sample) => self.on_presence(&sample),
                    Err(_) => break,
                },
                _ = tick.tick() => {
                    // A heartbeat counts from when it arrived (t2), so any
                    // still queued behind a busy host task goes in before
                    // leases and presence are judged.
                    while let Ok((id, heartbeat, t2)) = heartbeats.try_recv() {
                        self.on_heartbeat(id, heartbeat, t2);
                    }
                    if heard.judge(tokio::time::Instant::now()) {
                        let events = self.authority.expire(self.now());
                        self.record(events).await;
                    }
                    self.publish_ticks().await;
                    // Back from a stall, every edge's copy of its lease has
                    // run out by its own clock: each renewal the returning
                    // heartbeats bring goes out on the tick it arrives,
                    // not a quarter grace later.
                    self.publish_renewals(heard.listening()).await;
                    self.check_datagram().await;
                }
                _ = republish.tick() => self.publish().await,
                _ = health_out.tick() => self.publish_health().await,
                _ = health.tick() => {
                    let rows = self.health_rows();
                    if health_log.is_some() {
                        for row in &rows {
                            info!(target: "rlmesh_session::health", "{row}");
                        }
                    }
                    self.health_tx.send_replace(rows);
                    // A missed acknowledgement never detaches anyone: the
                    // step waits, and says for whom.
                    for step in self.auth.issuer().pending() {
                        info!(
                            robot = %step.robot,
                            version = step.version,
                            stage = step.stage,
                            waiting = ?step.waiting,
                            "stream key rotation pending",
                        );
                    }
                }
            }
            self.flush_keys().await;
        }
        debug!(session = %self.config.session, "session host stopped");
    }

    /// The next answer of a lobby verb that waited.
    #[cfg(feature = "fleet")]
    async fn lobby_answer(&mut self) -> Option<LobbyDone> {
        self.lobby_answers.recv().await
    }

    /// Without `fleet` there is no lobby, so no answer ever comes.
    #[cfg(not(feature = "fleet"))]
    async fn lobby_answer(&mut self) -> Option<std::convert::Infallible> {
        std::future::pending().await
    }

    #[cfg(not(feature = "fleet"))]
    async fn finish_lobby(&mut self, never: std::convert::Infallible) {
        match never {}
    }

    fn now(&self) -> i64 {
        self.config.clock.now_ns()
    }

    /// Logs and broadcasts events, keeps the link table in step with the
    /// roster, and publishes the new state.
    async fn record(&mut self, events: Vec<AuthorityEvent>) {
        if events.is_empty() {
            return;
        }
        for event in events {
            info!(session = %self.config.session, "{event}");
            self.warn_of(&event);
            match &event {
                AuthorityEvent::Attached { participant, .. } => {
                    self.log_data_path(participant);
                    let generation = self.declare_echo(participant).await;
                    let offset = self.config.offset;
                    // A new generation is a new process, on its own clock.
                    if self
                        .links
                        .get(participant)
                        .is_none_or(|link| link.generation != generation)
                    {
                        self.links.insert(
                            participant.clone(),
                            Link {
                                generation,
                                estimator: OffsetEstimator::new(offset),
                                stats: LinkStats::new(offset.window),
                                rejected: Vec::new(),
                                clock_refused: Vec::new(),
                                chunk_timing: Vec::new(),
                                chunks_accepted: 0,
                                datagram: None,
                            },
                        );
                    }
                }
                AuthorityEvent::Detached { participant, .. } => {
                    self.links.remove(participant);
                    // Its waiting lobby verbs end with it: the lobby sees
                    // their answers' receivers closed.
                    #[cfg(feature = "fleet")]
                    self.lobby.retain(|(id, _, _), waiting| {
                        let keep = id != participant;
                        if !keep {
                            waiting.task.abort();
                        }
                        keep
                    });
                    let failed = self.auth.issuer().detach(participant);
                    self.failed.extend(failed);
                    // Detaching ends the generation: its keys go with it.
                    if let Ok(mut peers) = self.peers.write()
                        && let Some(peer) = peers.current.remove(participant)
                    {
                        peers.by_key.retain(|_, p| !Arc::ptr_eq(p, &peer));
                    }
                }
                _ => {}
            }
            // Having no subscriber is fine.
            let _ = self.events.send(event);
        }
        let authority = &self.authority;
        self.auth
            .issuer()
            .end_tokens(|token| authority.leases().any(|l| l.token == token));
        self.sync_streams();
        self.sync_datagram().await;
        self.epoch
            .store(self.authority.epoch().0, Ordering::Relaxed);
        self.publish().await;
    }
}

impl HostActor {
    /// Warns of a participant the session dropped, or a robot held: each
    /// subject at most once per [`crate::quiet::QUIET`].
    fn warn_of(&mut self, event: &AuthorityEvent) {
        let session = &self.config.session;
        let now = std::time::Instant::now();
        match event {
            AuthorityEvent::Detached {
                participant,
                reason,
            } if *reason != DetachReason::Requested => {
                if let Some(held_back) = self.quiet.admit("dropped", participant.as_str(), now) {
                    warn!(%session, %participant, ?reason, held_back, "participant dropped from the session");
                }
            }
            AuthorityEvent::RobotHeld { robot, reason } => {
                if let Some(held_back) = self.quiet.admit("held", robot.as_str(), now) {
                    warn!(%session, %robot, reason = %reason.detail, code = ?reason.code, held_back, "robot held: it takes no lease until re-armed");
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stalled_host_listens_for_a_lease_grace_before_it_judges() {
        let ms = Duration::from_millis;
        let start = tokio::time::Instant::now();
        let mut heard = Heard::new(start, ms(10), ms(250));
        // On-time and merely late ticks judge.
        assert!(heard.judge(start + ms(10)));
        assert!(heard.judge(start + ms(100)));
        // The first tick after a stall does not, nor any within the grace.
        let back = start + ms(700);
        assert!(!heard.judge(back));
        assert!(!heard.judge(back + ms(10)));
        assert!(!heard.judge(back + ms(240)));
        // The judgement after the grace runs however late its tick is, so
        // a host that keeps stalling still expires what is silent.
        assert!(heard.judge(back + ms(900)));
        assert!(!heard.judge(back + ms(1500)));
        assert!(heard.judge(back + ms(1760)));
    }
}
