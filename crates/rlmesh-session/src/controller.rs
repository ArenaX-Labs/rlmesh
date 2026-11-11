//! The controller runtime: a bound policy's inputs, leases, chunks and run
//! loop. Language bindings wrap it; users write
//! `predict` and nothing else.
//!
//! - [`Controller::bind`] waits for the robot's parts, binds the policy
//!   (refuse-and-explain), declares the binding and opens its inputs
//!   ([`Inputs`]).
//! - **Leases.** A background keeper holds the binding's leases while they
//!   are wanted, as one [`LeaseGroup`] (the holder state machine the
//!   browser binding runs too, with its local deadline): it requests them, re-requests them after a preemption
//!   ends, a lapse or a re-arm (each with a fresh token), and declares a
//!   chunk sender per granted lease. On the wall clock it waits until the
//!   host has mapped this controller's clock and its robots' (an edge
//!   rejects chunks until then). [`Controller::acquire`] takes them at
//!   once (an operator taking over), once those clocks are mapped;
//!   [`Controller::release`] gives them back.
//! - [`Controller::send`] packs `predict`'s output per action entry
//!   (`ActionRow::pack`) and sends one timed chunk per commanded part:
//!   `t0` at the snapshot (or at send, `t0: "send"`), the binding's action
//!   grid, `valid_until = t0 + min(chunk × dt, ttl)`, `replace_from = t0`
//!   and the snapshot's `based_on`.
//! - [`Controller::run`] runs the policy loop: starts on the replan
//!   grid, one inference in flight, skipped starts with their reasons, warm
//!   before authority (leases are requested only once a first snapshot of
//!   the required inputs is complete), window gaps recovered from the
//!   host's history store, and reporting through [`StartSchedule`].
//!   On the wall clock the grid is phased just after the publication of
//!   one sampled input ([`PhaseAlign`]), so a snapshot reads a sample about
//!   a millisecond old rather than up to a full replan period old.
//!   If `predict` fails, the controller stops sending and releases its
//!   leases; the edge holds, then goes safe. If its robot is lost
//!   (`ROBOT_LOST`) or the host is (`HOST_LOST`), the run stops with
//!   [`RunError::Lost`]. In a scene's run a lost robot may rejoin: the loop
//!   keeps going without a lease (`no_lease`) and asks for its leases again
//!   once the run is re-armed; only the run closing stops it, and a run this
//!   process stopped returns. A run on a session that is not armed waits
//!   for it, and says so once each time the session leaves ARMED or ACTIVE
//!   ([`RunEvent::WaitingToArm`], [`Controller::waiting`]).
//! - **Rebinding.** A background watcher follows every robot the binding
//!   reads or commands, and the robots its selectors name. When one comes
//!   back, restarts, declares its parts again or appears, the controller
//!   binds the policy again against the session's current parts: a changed
//!   table is declared (the host re-issues the leases it holds under a fresh
//!   token, which the keeper takes over without asking again) and the inputs
//!   are opened afresh, so a restarted publisher's samples are taken from its
//!   first. An input whose publisher is gone reports it once
//!   ([`RunEvent::InputLost`]), its skipped starts count as
//!   `publisher_gone`, and coming back reports [`RunEvent::Rebound`]. If a
//!   part the binding uses comes back as something the policy cannot bind to,
//!   the run stops with [`RunError::Rebind`] naming it.
//!
//! [`Controller::run`] and [`Controller::send`] block: call them from a
//! plain thread, not from inside the async runtime.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::spec::apply::{ApplyError, Value};
use crate::spec::policy::ChunkStart;
use crate::spec::selector::Selector;
use crate::spec::{BindError, Binding, Policy, RunOptions};
use crate::wire::{
    AuthoritySnapshot, ChannelId, ClockMode, CtlReply, InputRange, Lease, Mode, PartSpec,
    ParticipantId, Reason, ReasonCode, RunCounters,
};
use bytes::Bytes;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::cache::{Inputs, Observation, ObserveError};
use crate::chunk::{ChunkDraft, ChunkPublisher};
use crate::error::SessionError;
use crate::lease::{LeaseChange, LeaseGroup};
use crate::participant::Participant;
use crate::report::run_counters;
use crate::schedule::{PhaseAlign, RunEvent, RunStats, SkipReason, StartSchedule};

/// Events kept for [`Controller::drain_events`].
const EVENTS: usize = 256;
/// How long the run loop sleeps at most between checks of its stop flag.
const IDLE_SLICE_NS: i64 = 20_000_000;
/// The least time between two history-store recoveries of a window gap.
const RECOVER_GAP_NS: i64 = 100_000_000;
/// The least time between two lease requests.
const REQUEST_GAP: Duration = Duration::from_millis(50);

/// How a controller runs.
#[derive(Clone, Debug)]
pub struct ControllerConfig {
    /// The lease ttl (`None`: the session's lease grace).
    pub lease_ttl: Option<Duration>,
    /// How long [`Controller::bind`] waits for the robot's parts, and
    /// [`Controller::acquire`] for the clocks to be mapped.
    pub bind_timeout: Duration,
    /// How often a wanted lease is re-requested while it is refused (it is
    /// also re-requested on every authority change).
    pub lease_retry: Duration,
    /// An inference running this long is reported.
    pub pending_after: Duration,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            lease_ttl: None,
            bind_timeout: Duration::from_secs(10),
            lease_retry: Duration::from_millis(250),
            pending_after: Duration::from_secs(2),
        }
    }
}

/// Why a controller could not bind.
#[derive(Debug, thiserror::Error)]
pub enum ControllerError {
    /// The binder refused (every problem) or the options do not fit.
    #[error(transparent)]
    Bind(#[from] BindError),
    /// The session failed.
    #[error(transparent)]
    Session(#[from] SessionError),
}

/// Why a chunk was not sent.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// `predict`'s output does not fit the binding.
    #[error("{0}")]
    Value(String),
    /// The session refused it.
    #[error(transparent)]
    Session(#[from] SessionError),
}

impl From<ApplyError> for SendError {
    fn from(err: ApplyError) -> Self {
        Self::Value(err.0)
    }
}

/// What happened to one action entry's chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sent {
    /// Sent with this seq.
    Seq(u64),
    /// Not sent: this controller does not hold the part's lease.
    NoLease,
    /// Not sent: the entry is dropped (`allow: drop`).
    Dropped,
}

/// Why [`Controller::run`] returned early.
#[derive(Debug, thiserror::Error)]
pub enum RunError<E> {
    /// `predict` (or the idle hook) failed; the leases were released.
    #[error("predict failed")]
    Predict(E),
    /// `predict`'s output did not fit the binding; the leases were
    /// released.
    #[error(transparent)]
    Send(SendError),
    /// The session is gone for this run: its robot's heartbeats lapsed
    /// (`ROBOT_LOST`) or the host's liveliness went away (`HOST_LOST`).
    #[error("the session was lost: {0}")]
    Lost(Reason),
    /// A part the binding uses came back as something the policy cannot
    /// bind to (another kind, shape or rate): the part, what changed and
    /// the binder's refusal. The leases were released.
    #[error("{0}")]
    Rebind(String),
}

/// The model side of [`Controller::run`].
pub trait Predictor {
    /// The model's error.
    type Error;

    /// One start: the observation (snapshotted at `snapshot_ns`, session
    /// time) in, one value per action entry out (`[chunk, ...]`).
    fn predict(
        &mut self,
        observation: Observation,
        snapshot_ns: i64,
    ) -> Result<Vec<(String, Value)>, Self::Error>;

    /// Called between starts at least every 20 ms (a binding checks for
    /// interrupts here). An error ends the run like a failed `predict`.
    fn idle(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// The lease keeper's shared state: the binding's leases as one
/// [`LeaseGroup`] (the holder state machine the browser binding runs too),
/// and a chunk sender per granted token.
struct Leases {
    group: Mutex<LeaseGroup>,
    publishers: Mutex<BTreeMap<ChannelId, ChunkPublisher>>,
    want: AtomicBool,
    /// Serialises grants and releases between the keeper and explicit
    /// calls.
    request: tokio::sync::Mutex<()>,
    wake: Notify,
    ttl: Option<Duration>,
}

impl Leases {
    fn group(&self) -> MutexGuard<'_, LeaseGroup> {
        self.group
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn publishers(&self) -> MutexGuard<'_, BTreeMap<ChannelId, ChunkPublisher>> {
        self.publishers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Applies the current authority view: renews the held leases, and
    /// drops what it no longer backs (lost to a preemption, lapsed, or
    /// voided by a new epoch) with its chunk sender.
    fn sync(&self, participant: &Participant) {
        let Some(snapshot) = participant.authority_view().current() else {
            return;
        };
        let now = participant.clock().now_ns();
        let changes =
            self.group()
                .on_snapshot(&snapshot, participant.clock_estimate().as_ref(), now);
        for (channel, change) in changes {
            if let LeaseChange::Revoked(reason) = change {
                let part = snapshot.part_name(channel);
                info!(participant = %participant.id(), %part, %reason, "lease lost");
                // A rebind may have taken the part over under a new token
                // meanwhile: only a sender of no held token goes.
                let held = self.group().lease(channel, now).map(|l| l.token);
                let mut publishers = self.publishers();
                if publishers
                    .get(&channel)
                    .is_some_and(|p| Some(p.token()) != held)
                {
                    publishers.remove(&channel);
                }
            }
        }
    }

    /// Takes over the leases the host re-issued for `binding` when it was
    /// declared (`state`, the reply): a lease group over its parts holding
    /// them, and a chunk sender per token. The parts not re-issued are
    /// left to the keeper.
    async fn rebind(
        &self,
        participant: &Participant,
        binding: &Binding,
        state: &AuthoritySnapshot,
    ) {
        let _request = self.request.lock().await;
        let me = participant.id();
        let channels = binding.channels();
        let now = participant.clock().now_ns();
        let estimate = participant.clock_estimate();
        let mut group = LeaseGroup::new(me, &channels, 0);
        for lease in state.leases.iter().filter(|l| {
            &l.holder == me && l.binding == binding.fingerprint && channels.contains(&l.channel)
        }) {
            // Sent a full ttl ago: the local deadline is the host expiry's
            // bound alone, never one this side made up.
            let ttl = i64::try_from(lease.ttl_ns).unwrap_or(i64::MAX);
            let _ = group.request_some(&[lease.channel], binding.fingerprint, None, now - ttl);
            let reply = CtlReply {
                result: Ok(()),
                state: state.clone(),
                leases: vec![lease.clone()],
                request_id: 0,
                keys: Vec::new(),
                assignment: None,
                run: None,
            };
            group.on_reply_some(&[lease.channel], &reply, estimate.as_ref(), now);
        }
        let held = group.held(now);
        *self.group() = group;
        self.publishers().retain(|channel, p| {
            held.iter()
                .any(|l| l.channel == *channel && l.token == p.token())
        });
        for lease in &held {
            if self.publishers().contains_key(&lease.channel) {
                continue;
            }
            match participant.chunk_publisher(lease).await {
                Ok(publisher) => {
                    let part = state.part_name(lease.channel);
                    info!(participant = %me, %part, token = %lease.token, "lease kept across a rebind");
                    self.publishers().insert(lease.channel, publisher);
                }
                Err(err) => warn!(participant = %me, %err, "no chunk sender after a rebind"),
            }
        }
    }

    fn holds_all(&self, participant: &Participant) -> bool {
        self.group().holds_all(participant.clock().now_ns())
    }

    /// Whether the host has mapped this controller's clock and the clock of
    /// every present robot it commands (see [`unmapped_clocks`]).
    fn clocks_mapped(&self, participant: &Participant) -> bool {
        let channels = self.group().channels();
        participant.authority_view().with(|view| {
            view.is_some_and(|view| unmapped_clocks(view, participant.id(), &channels).is_empty())
        })
    }

    /// Waits until the clocks are mapped, or the session stops being armed
    /// (the grant then says why).
    async fn wait_for_clocks(
        &self,
        participant: &Participant,
        timeout: Duration,
    ) -> Result<(), SessionError> {
        let channels = self.group().channels();
        let me = participant.id().clone();
        let what = "the session host to measure the clocks of this controller and its robots";
        let waited = participant
            .wait_for(what, timeout, |view| {
                !matches!(view.mode, Mode::Armed | Mode::Active)
                    || unmapped_clocks(view, &me, &channels).is_empty()
            })
            .await;
        match waited {
            Err(SessionError::Timeout(timeout, _)) => {
                let unmapped = participant.authority().map_or_else(Vec::new, |view| {
                    unmapped_clocks(&view, &me, &channels)
                        .iter()
                        .map(ToString::to_string)
                        .collect()
                });
                Err(SessionError::Timeout(
                    timeout,
                    format!(
                        "the session host to measure the clock of {}: an edge rejects chunks \
                         until it has",
                        unmapped.join(", ")
                    ),
                ))
            }
            other => other.map(|_| ()),
        }
    }

    /// Requests the leases not held, robot by robot, so a robot that is
    /// gone or stopped on its own never keeps the others from moving, and
    /// declares a sender per new token. Fails only when no robot granted
    /// any.
    async fn grant(&self, participant: &Participant) -> Result<Vec<Lease>, SessionError> {
        let _request = self.request.lock().await;
        let ttl_ns = self
            .ttl
            .map(|ttl| u64::try_from(ttl.as_nanos()).unwrap_or(u64::MAX));
        let unheld = self.group().unheld(participant.clock().now_ns());
        // One request per robot, for its present parts.
        let mut robots: Vec<(ParticipantId, Vec<ChannelId>)> = Vec::new();
        participant.authority_view().with(|view| {
            let Some(view) = view else {
                return;
            };
            for channel in &unheld {
                let Some(part) = view.part(*channel) else {
                    continue;
                };
                if !view.participant(&part.owner).is_some_and(|p| p.present) {
                    continue;
                }
                match robots.iter_mut().find(|(r, _)| *r == part.owner) {
                    Some((_, channels)) => channels.push(*channel),
                    None => robots.push((part.owner.clone(), vec![*channel])),
                }
            }
        });
        let mut granted = Vec::new();
        let mut refused = None;
        for (_, channels) in robots {
            let command = self.group().request_some(
                &channels,
                participant.binding_fingerprint(),
                ttl_ns,
                participant.clock().now_ns(),
            );
            let reply = match participant.command(command).await {
                Ok(reply) => reply,
                Err(err) => {
                    refused = Some(err);
                    continue;
                }
            };
            self.group().on_reply_some(
                &channels,
                &reply,
                participant.clock_estimate().as_ref(),
                participant.clock().now_ns(),
            );
            match &reply.result {
                Ok(()) => granted.extend(reply.leases.iter().cloned()),
                Err(err) => refused = Some(SessionError::Rejected(err.clone())),
            }
        }
        if granted.is_empty()
            && let Some(err) = refused
        {
            return Err(err);
        }
        for lease in &granted {
            let current = self
                .publishers()
                .get(&lease.channel)
                .is_some_and(|p| p.token() == lease.token);
            if current {
                continue;
            }
            let publisher = participant.chunk_publisher(lease).await?;
            let part = participant
                .authority()
                .map_or_else(|| lease.channel.to_string(), |v| v.part_name(lease.channel));
            info!(participant = %participant.id(), %part, token = %lease.token, "lease granted");
            self.publishers().insert(lease.channel, publisher);
        }
        Ok(granted)
    }

    /// Gives every held lease back.
    async fn release(&self, participant: &Participant) -> Result<(), SessionError> {
        let _request = self.request.lock().await;
        let commands = self.group().release();
        self.publishers().clear();
        for command in commands {
            // A refusal means it is already gone (preempted, lapsed, a new
            // epoch): nothing to give back.
            participant.command(command).await?;
        }
        Ok(())
    }
}

/// Who still needs a clock mapping before `me` can command `channels`: an
/// edge rejects a chunk until its robot's clock and its sender's are mapped
/// (`NoClockMapping`), so leases taken earlier only produce rejected chunks.
/// A ticked session needs none, and a robot that is gone is asked for no
/// lease, so it never keeps the others waiting.
fn unmapped_clocks<'a>(
    view: &'a AuthoritySnapshot,
    me: &'a ParticipantId,
    channels: &[ChannelId],
) -> Vec<&'a ParticipantId> {
    if view.clock_mode != ClockMode::Wall {
        return Vec::new();
    }
    let mut unmapped: Vec<&ParticipantId> = Vec::new();
    if view.clock(me).is_none() {
        unmapped.push(me);
    }
    for part in channels.iter().filter_map(|c| view.part(*c)) {
        if view.clock(&part.owner).is_none()
            && view.participant(&part.owner).is_some_and(|p| p.present)
            && !unmapped.contains(&&part.owner)
        {
            unmapped.push(&part.owner);
        }
    }
    unmapped
}

/// Keeps the leases while they are wanted (see the module docs).
async fn keep_leases(
    participant: Arc<Participant>,
    leases: Arc<Leases>,
    retry: Duration,
    cancel: CancellationToken,
) {
    let mut view = participant.watch();
    loop {
        leases.sync(&participant);
        let wanted = leases.want.load(Ordering::Relaxed);
        let mode = participant.authority().map(|v| v.mode);
        if wanted
            && !leases.holds_all(&participant)
            && matches!(mode, Some(Mode::Armed | Mode::Active))
            && leases.clocks_mapped(&participant)
        {
            match leases.grant(&participant).await {
                Ok(_) => {}
                Err(err) => {
                    debug!(participant = %participant.id(), %err, "lease request refused");
                    tokio::select! {
                        () = cancel.cancelled() => return,
                        () = tokio::time::sleep(REQUEST_GAP) => {}
                    }
                }
            }
        }
        tokio::select! {
            () = cancel.cancelled() => return,
            changed = view.changed() => if changed.is_err() { return },
            () = leases.wake.notified() => {}
            () = tokio::time::sleep(retry) => {}
        }
    }
}

/// Reports an inference pending past the configured bound (a timer,
/// not an average over completions).
async fn watch_pending(
    participant: Arc<Participant>,
    schedule: Arc<Mutex<StartSchedule>>,
    events: Arc<Mutex<VecDeque<RunEvent>>>,
    after: Duration,
    cancel: CancellationToken,
) {
    let mut reported_start = None;
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            _ = tick.tick() => {}
        }
        let (pending, starts) = {
            let schedule = schedule
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                schedule.pending(participant.clock().now_ns()),
                schedule.stats().starts,
            )
        };
        if let Some(pending) = pending.filter(|p| *p >= after)
            && reported_start != Some(starts)
        {
            reported_start = Some(starts);
            push_event(&events, RunEvent::InferencePending { pending });
        }
    }
}

/// A binding and the inputs opened for it, swapped together when the
/// controller binds again.
struct Bound {
    binding: Arc<Binding>,
    inputs: Inputs,
}

/// One robot a binding reads, commands or names, as the authority state
/// shows it: present or not, its process (generation) and its parts'
/// streams.
#[derive(Clone, Debug, PartialEq)]
struct Source {
    present: bool,
    generation: u32,
    parts: Vec<(String, ChannelId)>,
}

impl Source {
    /// `robot` in `view`; `None` when the session has never heard of it.
    fn of(view: &AuthoritySnapshot, robot: &ParticipantId) -> Option<Self> {
        let info = view.participant(robot);
        let parts: Vec<(String, ChannelId)> = view
            .parts
            .iter()
            .filter(|p| &p.owner == robot)
            .map(|p| (p.path.to_string(), p.channel))
            .collect();
        if info.is_none() && parts.is_empty() {
            return None;
        }
        Some(Self {
            present: info.is_some_and(|i| i.present),
            generation: info.map_or(0, |i| i.generation),
            parts,
        })
    }
}

/// An input whose publisher is gone.
#[derive(Clone, Debug, PartialEq)]
struct Gone {
    entry: String,
    part: String,
    why: String,
}

/// What the watcher knows of the binding's sources.
#[derive(Default)]
struct Sources {
    /// Each watched robot as it was last looked at.
    seen: BTreeMap<ParticipantId, Option<Source>>,
    /// The inputs whose publisher is gone.
    gone: Vec<Gone>,
    /// Why the binding no longer stands: ends a run.
    failed: Option<String>,
}

/// Binds the policy again when its sources change (see the module docs).
struct Rebinder {
    policy: Policy,
    options: RunOptions,
    bound: Mutex<Arc<Bound>>,
    sources: Mutex<Sources>,
}

impl Rebinder {
    fn bound(&self) -> Arc<Bound> {
        self.bound
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn sources(&self) -> MutexGuard<'_, Sources> {
        self.sources
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The robots to follow: those `binding` reads or commands, the robots
    /// the policy's selectors (and the run's `at` overrides) name, and the
    /// robot it was bound to.
    fn watched(&self, binding: &Binding) -> BTreeSet<ParticipantId> {
        let mut robots = BTreeSet::new();
        robots.insert(binding.robot.clone());
        robots.extend(binding.places.iter().map(|p| p.robot.clone()));
        let parts = binding
            .inputs
            .iter()
            .filter_map(|r| r.part.as_ref())
            .chain(binding.actions.iter().filter_map(|r| r.part.as_ref()));
        robots.extend(parts.map(|p| p.owner.clone()));
        let entries = self
            .policy
            .inputs
            .iter()
            .map(|i| &i.entry)
            .chain(self.policy.actions.iter().map(|a| &a.entry));
        for entry in entries {
            let selector = match self.options.at.get(&entry.name) {
                Some(path) => Some(Selector::literal(path)),
                None => entry.at.clone(),
            };
            robots.extend(selector.and_then(|s| s.robot));
        }
        robots
    }

    fn look(
        &self,
        participant: &Participant,
        binding: &Binding,
    ) -> Option<BTreeMap<ParticipantId, Option<Source>>> {
        let watched = self.watched(binding);
        participant.authority_view().with(|view| {
            let view = view?;
            Some(
                watched
                    .into_iter()
                    .map(|robot| {
                        let source = Source::of(view, &robot);
                        (robot, source)
                    })
                    .collect(),
            )
        })
    }

    /// Takes the sources as they are now as the ones bound against.
    fn settle(&self, participant: &Participant) {
        let bound = self.bound();
        if let Some(seen) = self.look(participant, &bound.binding) {
            self.sources().seen = seen;
        }
    }

    /// Looks at the sources again: reports inputs whose publisher went away,
    /// and binds again when one came back, restarted, declared its parts
    /// again or appeared.
    async fn check(
        &self,
        participant: &Participant,
        leases: &Leases,
        events: &Mutex<VecDeque<RunEvent>>,
    ) {
        let before = {
            let sources = self.sources();
            if sources.failed.is_some() {
                return;
            }
            sources.seen.clone()
        };
        let bound = self.bound();
        let Some(now) = self.look(participant, &bound.binding) else {
            return;
        };
        if now == before {
            return;
        }
        let gone_before = participant.authority_view().with(|v| {
            v.map(|v| gone_inputs(v, &bound.binding))
                .unwrap_or_default()
        });
        let changed: Vec<ParticipantId> = now
            .iter()
            .filter(|(robot, source)| before.get(*robot) != Some(*source))
            .map(|(robot, _)| robot.clone())
            .collect();
        let rebound = match self
            .rebind(participant, leases, &bound, &before, &now, &changed)
            .await
        {
            Ok(rebound) => rebound,
            Err(Failed::Transient(err)) => {
                // Looked at again on the next view.
                warn!(participant = %participant.id(), %err, "binding again failed; retrying");
                return;
            }
            Err(Failed::Stands(why)) => {
                warn!(participant = %participant.id(), "{why}");
                self.sources().failed = Some(why);
                return;
            }
        };
        let current = self.bound();
        let gone_now = participant.authority_view().with(|v| {
            v.map(|v| gone_inputs(v, &current.binding))
                .unwrap_or_default()
        });
        let mut sources = self.sources();
        sources.seen = now;
        let reported: Vec<String> = sources.gone.iter().map(|g| g.entry.clone()).collect();
        // A part declared again in one step is bound again at once: only
        // what stays gone, or went blank, was lost.
        for lost in gone_before.iter().filter(|g| !reported.contains(&g.entry)) {
            let blank = current
                .binding
                .input(&lost.entry)
                .is_some_and(|row| row.part.is_none());
            if !blank && !gone_now.iter().any(|g| g.entry == lost.entry) {
                continue;
            }
            let meanwhile = if blank {
                "it reads blank until it is back".to_string()
            } else {
                format!(
                    "starts that need it are skipped ({}) until it is back",
                    SkipReason::PublisherGone
                )
            };
            push_event(
                events,
                RunEvent::InputLost {
                    part: lost.part.clone(),
                    message: format!(
                        "input {:?} lost its publisher: {}; {meanwhile}",
                        lost.entry, lost.why
                    ),
                },
            );
        }
        sources.gone = gone_now;
        drop(sources);
        if let Some(event) = rebound {
            push_event(events, event);
        }
    }

    /// Binds the policy against the current parts when a robot it follows
    /// changed; declares the table when it changed, and opens the inputs
    /// afresh when a part moved or a robot it reads restarted. The event to
    /// report, if it bound again.
    async fn rebind(
        &self,
        participant: &Participant,
        leases: &Leases,
        bound: &Bound,
        before: &BTreeMap<ParticipantId, Option<Source>>,
        now: &BTreeMap<ParticipantId, Option<Source>>,
        changed: &[ParticipantId],
    ) -> Result<Option<RunEvent>, Failed> {
        let old = &bound.binding;
        let next = match participant.bind(&self.policy, &self.options) {
            Ok(next) => next,
            Err(err) if err.waiting_for().is_some() => return Ok(None),
            Err(err) => {
                let uses = rows(old).any(|(_, part)| changed.contains(&part.owner));
                if !uses {
                    warn!(participant = %participant.id(), %err, "a robot the policy names cannot be bound");
                    return Ok(None);
                }
                let changes = participant.authority_view().with(|view| {
                    view.map(|view| what_changed(view, old, changed))
                        .unwrap_or_default()
                });
                return Err(Failed::Stands(format!(
                    "{changes}: policy {} no longer binds, so its run stopped.\n{err}",
                    self.options.name
                )));
            }
        };
        // A robot it reads whose process changed: its seqs start over, so
        // its streams are subscribed afresh.
        let restarted: Vec<&ParticipantId> = changed
            .iter()
            .filter(|robot| {
                let generation = |s: &BTreeMap<ParticipantId, Option<Source>>| {
                    s.get(*robot)
                        .cloned()
                        .flatten()
                        .map(|s| (s.present, s.generation))
                };
                generation(now).is_some_and(|(present, _)| present)
                    && generation(before) != generation(now)
            })
            .collect();
        let reads_restarted = rows(&next).any(|(_, part)| restarted.contains(&&part.owner));
        if next.fingerprint == old.fingerprint && !reads_restarted {
            return Ok(None);
        }
        if next.fingerprint != old.fingerprint {
            let state = participant
                .declare_bound(&next)
                .await
                .map_err(|err| match err {
                    SessionError::Rejected(refusal) => Failed::Stands(format!(
                        "the session refused policy {}'s binding again: {}",
                        self.options.name, refusal.message
                    )),
                    other => Failed::Transient(other),
                })?;
            leases.rebind(participant, &next, &state).await;
        }
        let inputs = Inputs::open(participant, &next)
            .await
            .map_err(Failed::Transient)?;
        let message = rebound_message(old, &next, &restarted, before);
        let parts = rebound_parts(old, &next, &restarted);
        *self
            .bound
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::new(Bound {
            binding: Arc::new(next),
            inputs,
        });
        info!(participant = %participant.id(), "{message}");
        Ok((!parts.is_empty()).then_some(RunEvent::Rebound { parts, message }))
    }
}

/// Why binding again did not happen.
enum Failed {
    /// Tried again on the next view.
    Transient(SessionError),
    /// The binding no longer stands.
    Stands(String),
}

/// Every part a binding reads or commands, by entry.
fn rows(binding: &Binding) -> impl Iterator<Item = (&str, &PartSpec)> {
    binding
        .inputs
        .iter()
        .filter_map(|r| Some((r.entry.as_str(), r.part.as_ref()?)))
        .chain(
            binding
                .actions
                .iter()
                .filter_map(|r| Some((r.entry.as_str(), r.part.as_ref()?))),
        )
}

/// The inputs of `binding` whose publisher `view` shows gone: absent, or no
/// longer declaring the part.
fn gone_inputs(view: &AuthoritySnapshot, binding: &Binding) -> Vec<Gone> {
    binding
        .inputs
        .iter()
        .filter_map(|row| {
            let part = row.part.as_ref()?;
            let owner = &part.owner;
            let why = match view.participant(owner) {
                Some(info) if info.present => {
                    if view.parts.iter().any(|p| p.channel == part.channel) {
                        return None;
                    }
                    format!("{owner} no longer declares {}", part.path)
                }
                Some(info) => match &info.left {
                    Some(reason) => format!("{owner} left the session ({reason})"),
                    None => format!("{owner} left the session"),
                },
                None => format!("{owner} left the session"),
            };
            Some(Gone {
                entry: row.entry.clone(),
                part: part.qualified(),
                why,
            })
        })
        .collect()
}

/// How the parts `old` binds on the `changed` robots differ in `view`:
/// `robot:path` and what changed, for a refusal.
fn what_changed(view: &AuthoritySnapshot, old: &Binding, changed: &[ParticipantId]) -> String {
    let mut seen: Vec<&str> = Vec::new();
    let mut said = Vec::new();
    for (_, part) in rows(old).filter(|(_, p)| changed.contains(&p.owner)) {
        if seen.contains(&part.path.as_str()) {
            continue;
        }
        seen.push(part.path.as_str());
        let now = view
            .parts
            .iter()
            .find(|p| p.owner == part.owner && p.path == part.path);
        let Some(now) = now else {
            said.push(format!("{} is no longer declared", part.qualified()));
            continue;
        };
        let mut diffs = Vec::new();
        if now.kind != part.kind {
            diffs.push(format!("kind {}, now {}", part.kind, now.kind));
        }
        let fields = |p: &PartSpec| -> Vec<String> {
            p.fields
                .iter()
                .map(|f| format!("{} {:?}{:?}", f.name, f.dtype, f.shape))
                .collect()
        };
        if fields(now) != fields(part) {
            diffs.push(format!(
                "fields {}, now {}",
                fields(part).join(", "),
                fields(now).join(", ")
            ));
        }
        if now.hz.to_bits() != part.hz.to_bits() {
            diffs.push(format!("{} Hz, now {} Hz", part.hz, now.hz));
        }
        if now.names != part.names {
            diffs.push(format!("names {:?}, now {:?}", part.names, now.names));
        }
        if now.command != part.command {
            diffs.push("its command, changed".into());
        }
        if diffs.is_empty() && now.channel != part.channel {
            diffs.push("declared again".into());
        }
        if !diffs.is_empty() {
            said.push(format!(
                "{} came back changed ({})",
                part.qualified(),
                diffs.join("; ")
            ));
        }
    }
    if said.is_empty() {
        let robots: Vec<String> = changed.iter().map(ToString::to_string).collect();
        return format!("{} declared other parts", robots.join(", "));
    }
    said.join("; ")
}

/// The parts bound again: those that moved to another stream, those a
/// restarted robot publishes, and the inputs and actions that bound where
/// they were blank or dropped.
fn rebound_parts(old: &Binding, next: &Binding, restarted: &[&ParticipantId]) -> Vec<String> {
    let mut parts = Vec::new();
    for (entry, part) in rows(next) {
        let before = rows(old).find(|(e, _)| *e == entry).map(|(_, p)| p);
        let moved = before.is_none_or(|b| b.channel != part.channel || b.owner != part.owner);
        if (moved || restarted.contains(&&part.owner)) && !parts.contains(&part.qualified()) {
            parts.push(part.qualified());
        }
    }
    parts
}

/// What binding again did, in words.
fn rebound_message(
    old: &Binding,
    next: &Binding,
    restarted: &[&ParticipantId],
    before: &BTreeMap<ParticipantId, Option<Source>>,
) -> String {
    let mut said = Vec::new();
    for (entry, part) in rows(next) {
        let was = rows(old).find(|(e, _)| *e == entry).map(|(_, p)| p);
        let owner = &part.owner;
        let line = match was {
            None => format!(
                "{entry:?} reads {} now that {owner} is in the session",
                part.qualified()
            ),
            Some(was) if was.channel != part.channel || was.owner != part.owner => {
                format!(
                    "{entry:?} is bound again to {}, declared again",
                    part.qualified()
                )
            }
            Some(_) if restarted.contains(&owner) => {
                let back = before
                    .get(owner)
                    .cloned()
                    .flatten()
                    .is_some_and(|s| !s.present);
                let how = if back { "is back" } else { "restarted" };
                format!("{entry:?} reads {} again: {owner} {how}", part.qualified())
            }
            Some(_) => continue,
        };
        said.push(line);
    }
    for row in next.inputs.iter().filter(|r| r.part.is_none()) {
        if old.input(&row.entry).is_some_and(|r| r.part.is_some()) {
            said.push(format!("{:?} is blank now", row.entry));
        }
    }
    if said.is_empty() {
        return format!("policy {} bound again", next.policy);
    }
    format!("policy {} bound again: {}", next.policy, said.join("; "))
}

/// Follows the binding's sources until cancelled (see the module docs).
async fn watch_sources(
    participant: Arc<Participant>,
    rebinder: Arc<Rebinder>,
    leases: Arc<Leases>,
    events: Arc<Mutex<VecDeque<RunEvent>>>,
    cancel: CancellationToken,
) {
    let mut view = participant.watch();
    loop {
        rebinder.check(&participant, &leases, &events).await;
        tokio::select! {
            () = cancel.cancelled() => return,
            changed = view.changed() => if changed.is_err() { return },
        }
    }
}

fn push_event(events: &Mutex<VecDeque<RunEvent>>, event: RunEvent) {
    warn!(%event, "controller");
    let mut events = events
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if events.len() == EVENTS {
        events.pop_front();
    }
    events.push_back(event);
}

/// A bound policy: see the module docs.
pub struct Controller {
    participant: Arc<Participant>,
    rebinder: Arc<Rebinder>,
    leases: Arc<Leases>,
    schedule: Arc<Mutex<StartSchedule>>,
    events: Arc<Mutex<VecDeque<RunEvent>>>,
    runtime: tokio::runtime::Handle,
    /// How long [`Self::acquire`] waits for the clocks to be mapped.
    clock_timeout: Duration,
    stop: AtomicBool,
    /// The run loop's current phase offset (ns after the replan grid).
    phase_offset_ns: AtomicI64,
    /// Bumped by every run, so a stop request reaches only the run it was
    /// meant for.
    runs: AtomicU64,
    cancel: CancellationToken,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

impl Controller {
    /// Waits for `robot`'s parts, binds `policy` to them, declares the
    /// binding and opens its inputs. Only a robot that has not appeared yet
    /// is waited for (a [`SessionError::Timeout`] naming it, past the bind
    /// timeout); any other refusal returns at once. Call it on a Tokio
    /// runtime: the controller's background tasks run there.
    pub async fn bind(
        participant: Arc<Participant>,
        policy: &Policy,
        options: &RunOptions,
        config: ControllerConfig,
    ) -> Result<Self, ControllerError> {
        let robot: Option<ParticipantId> = options.robot.clone();
        let me = participant.id().clone();
        let deadline = tokio::time::Instant::now() + config.bind_timeout;
        let first = robot.as_ref().map_or_else(
            || "a robot to declare its parts".to_string(),
            |r| format!("robot {r} to declare its parts"),
        );
        participant
            .wait_for(&first, config.bind_timeout, |view| {
                view.parts
                    .iter()
                    .any(|p| p.owner != me && robot.as_ref().is_none_or(|r| &p.owner == r))
            })
            .await?;
        // Robots declare one by one: a binding that names robots not
        // declared yet is tried again on each new view until it stands or
        // the timeout passes. Anything else it is refused for stands.
        let mut view = participant.watch();
        let binding = loop {
            match participant.bind(policy, options) {
                Ok(binding) => break binding,
                Err(err) => {
                    let Some(waiting) = err.waiting_for() else {
                        return Err(err.into());
                    };
                    let left = deadline.saturating_duration_since(tokio::time::Instant::now());
                    match tokio::time::timeout(left, view.changed()).await {
                        Ok(Ok(())) => {}
                        Ok(Err(_)) => return Err(SessionError::Closed.into()),
                        Err(_) => {
                            return Err(SessionError::Timeout(config.bind_timeout, waiting).into());
                        }
                    }
                }
            }
        };
        participant.declare_bound(&binding).await?;
        let inputs = Inputs::open(&participant, &binding).await?;
        let rebinder = Arc::new(Rebinder {
            policy: policy.clone(),
            options: options.clone(),
            bound: Mutex::new(Arc::new(Bound {
                binding: Arc::new(binding.clone()),
                inputs,
            })),
            sources: Mutex::new(Sources::default()),
        });
        rebinder.settle(&participant);
        let leases = Arc::new(Leases {
            group: Mutex::new(LeaseGroup::new(participant.id(), &binding.channels(), 0)),
            publishers: Mutex::new(BTreeMap::new()),
            want: AtomicBool::new(false),
            request: tokio::sync::Mutex::new(()),
            wake: Notify::new(),
            ttl: config.lease_ttl,
        });
        let schedule = Arc::new(Mutex::new(StartSchedule::new(binding.replan_hz)));
        participant.set_run_source(Some({
            let schedule = schedule.clone();
            Arc::new(move |now_ns| {
                schedule.lock().map_or_else(
                    |_| RunCounters::default(),
                    |s| run_counters(&s.stats(), s.pending(now_ns)),
                )
            })
        }));
        let events = Arc::new(Mutex::new(VecDeque::new()));
        let cancel = CancellationToken::new();
        let tasks = vec![
            tokio::spawn(keep_leases(
                participant.clone(),
                leases.clone(),
                config.lease_retry,
                cancel.clone(),
            )),
            tokio::spawn(watch_pending(
                participant.clone(),
                schedule.clone(),
                events.clone(),
                config.pending_after,
                cancel.clone(),
            )),
            tokio::spawn(watch_sources(
                participant.clone(),
                rebinder.clone(),
                leases.clone(),
                events.clone(),
                cancel.clone(),
            )),
        ];
        Ok(Self {
            participant,
            rebinder,
            leases,
            schedule,
            events,
            runtime: tokio::runtime::Handle::current(),
            clock_timeout: config.bind_timeout,
            stop: AtomicBool::new(false),
            phase_offset_ns: AtomicI64::new(0),
            runs: AtomicU64::new(0),
            cancel,
            tasks,
        })
    }

    /// Gives the leases back, stops the background tasks and hands the
    /// participant back (to detach it).
    pub async fn close(mut self) -> Arc<Participant> {
        self.leases.want.store(false, Ordering::Relaxed);
        if let Err(err) = self.leases.release(&self.participant).await {
            warn!(%err, "releasing the leases on close failed");
        }
        self.cancel.cancel();
        for task in self.tasks.drain(..) {
            let _ = task.await;
        }
        self.participant.clone()
    }

    /// The binding table in force (it changes when the controller binds
    /// again).
    pub fn binding(&self) -> Arc<Binding> {
        self.rebinder.bound().binding.clone()
    }

    /// The participant.
    pub fn participant(&self) -> &Arc<Participant> {
        &self.participant
    }

    /// A policy start now: the snapshot (session time), each input through
    /// its plan, and the session time it was taken at.
    pub fn observe(&self) -> Result<(Observation, i64), ObserveError> {
        let now = self.participant.session_now_ns();
        self.rebinder.bound().inputs.observe(now).map(|o| (o, now))
    }

    /// Whether this controller holds every lease its binding commands.
    pub fn holds(&self) -> bool {
        self.leases.sync(&self.participant);
        self.leases.holds_all(&self.participant)
    }

    /// Whether it holds at least one of its leases.
    pub fn holds_any(&self) -> bool {
        !self.leases().is_empty()
    }

    /// The leases held now.
    pub fn leases(&self) -> Vec<Lease> {
        self.leases.sync(&self.participant);
        self.leases.group().held(self.participant.clock().now_ns())
    }

    /// Each robot this controller commands that is held, with why it takes
    /// no lease and how to re-arm it, by the authority state: no request to
    /// the host.
    pub fn held_refusals(&self) -> Vec<(ParticipantId, String)> {
        let channels = self.binding().channels();
        self.participant.authority_view().with(|view| {
            let Some(view) = view else {
                return Vec::new();
            };
            let mut robots: Vec<&ParticipantId> = channels
                .iter()
                .filter_map(|channel| view.part(*channel))
                .map(|part| &part.owner)
                .collect();
            robots.sort();
            robots.dedup();
            robots
                .into_iter()
                .filter_map(|robot| Some((robot.clone(), view.held_refusal(robot)?)))
                .collect()
        })
    }

    /// What this controller waits for before it can command, when the
    /// session is not armed: its mode in words. `None` while armed or
    /// active, and for a policy that commands nothing.
    pub fn waiting(&self) -> Option<String> {
        if self.binding().channels().is_empty() {
            return None;
        }
        let mode = self
            .participant
            .authority_view()
            .with(|v| v.map(|v| v.mode))?;
        let arm = "robot.arm(), or Arm in the cockpit";
        match mode {
            Mode::Armed | Mode::Active => None,
            Mode::Disarmed => Some(format!(
                "the session is DISARMED: this policy commands nothing until it is armed ({arm})"
            )),
            Mode::Safe => Some(format!(
                "the session is SAFE: this policy commands nothing until it is armed again ({arm})"
            )),
            Mode::Estop => Some(
                "the session is in ESTOP: this policy commands nothing until the e-stop is \
                 cleared and the session disarmed and armed again"
                    .into(),
            ),
        }
    }

    /// Reports what the run waits for, once each time it changes.
    fn report_waiting(&self, reported: &mut Option<String>) {
        let now = self.waiting();
        if let Some(message) = &now
            && reported.as_ref() != Some(message)
        {
            push_event(
                &self.events,
                RunEvent::WaitingToArm {
                    message: message.clone(),
                },
            );
        }
        *reported = now;
    }

    /// Reports each robot held since the last look (`reported`).
    fn report_held(&self, reported: &mut Vec<ParticipantId>) {
        let now = self.held_refusals();
        for (robot, message) in &now {
            if !reported.contains(robot) {
                warn!(participant = %self.participant.id(), "{message}");
                push_event(
                    &self.events,
                    RunEvent::RobotHeld {
                        message: message.clone(),
                    },
                );
            }
        }
        *reported = now.into_iter().map(|(robot, _)| robot).collect();
    }

    /// Each leased part whose edge refuses this controller's chunks for its
    /// clock, by the authority state: the part and the refusal in words.
    pub fn clock_refusals(&self) -> Vec<(ChannelId, String)> {
        let leases = self.leases();
        let me = self.participant.id();
        self.participant.authority_view().with(|view| {
            let Some(view) = view else {
                return Vec::new();
            };
            leases
                .iter()
                .filter_map(|lease| {
                    let refusal = view.clock_refusal(lease.channel, me)?;
                    let part = view.part(lease.channel)?;
                    Some((
                        lease.channel,
                        refusal.explain(&part.owner, part.path.as_str()),
                    ))
                })
                .collect()
        })
    }

    /// Reports each part that began refusing for its clock since the last
    /// look (`refused`).
    fn report_clock_refusals(&self, refused: &mut Vec<ChannelId>) {
        let now = self.clock_refusals();
        for (channel, message) in &now {
            if !refused.contains(channel) {
                warn!(participant = %self.participant.id(), "{message}");
                push_event(
                    &self.events,
                    RunEvent::ClockRefused {
                        message: message.clone(),
                    },
                );
            }
        }
        *refused = now.into_iter().map(|(channel, _)| channel).collect();
    }

    /// Takes the binding's leases now (all or none), preempting a lower
    /// class; from then on the keeper holds them. First waits for the host
    /// to map the clocks an edge needs (a [`SessionError::Timeout`] naming
    /// whose is missing, past the bind timeout), so a chunk sent right after
    /// is not rejected. A refusal (`LEASE_BUSY`, `GROUP_BUSY`, `NOT_READY`,
    /// ...) leaves the keeper as it was.
    pub async fn acquire(&self) -> Result<Vec<Lease>, SessionError> {
        self.leases
            .wait_for_clocks(&self.participant, self.clock_timeout)
            .await?;
        let leases = self.leases.grant(&self.participant).await?;
        self.leases.want.store(true, Ordering::Relaxed);
        Ok(leases)
    }

    /// Gives the leases back and stops the keeper from re-requesting them.
    pub async fn release(&self) -> Result<(), SessionError> {
        self.leases.want.store(false, Ordering::Relaxed);
        self.leases.release(&self.participant).await
    }

    /// Lets the keeper request the leases (once warm) and keep them.
    pub fn want_leases(&self) {
        if !self.leases.want.swap(true, Ordering::Relaxed) {
            self.leases.wake.notify_one();
        }
    }

    /// Packs `values` (one per action entry) and sends one chunk per
    /// commanded part, computed from the snapshot taken at `snapshot_ns`
    /// (session time) from the inputs `based_on`.
    pub fn send(
        &self,
        values: &[(String, Value)],
        based_on: &[InputRange],
        snapshot_ns: i64,
    ) -> Result<Vec<(String, Sent)>, SendError> {
        let binding = self.binding();
        if let Some((entry, _)) = values
            .iter()
            .find(|(entry, _)| binding.action(entry).is_none())
        {
            return Err(SendError::Value(format!(
                "{entry:?} is not an action of this policy (its actions: {})",
                action_names(&binding)
            )));
        }
        let mut sent = Vec::with_capacity(binding.actions.len());
        for row in &binding.actions {
            let Some(part) = &row.part else {
                sent.push((row.entry.clone(), Sent::Dropped));
                continue;
            };
            let value = values
                .iter()
                .find(|(entry, _)| *entry == row.entry)
                .map(|(_, v)| v)
                .ok_or_else(|| {
                    SendError::Value(format!(
                        "no value for action {:?} (expected {})",
                        row.entry, row.delivers
                    ))
                })?;
            let actions = Bytes::from(row.pack(value)?);
            let t0_ns = match row.t0 {
                ChunkStart::Snapshot => snapshot_ns,
                ChunkStart::Send => self.participant.session_now_ns(),
            };
            let draft = ChunkDraft {
                based_on: based_on.to_vec(),
                t0_ns,
                dt_ns: u64::try_from(row.dt.as_nanos()).unwrap_or(u64::MAX),
                replace_from_ns: t0_ns,
                valid_until_ns: t0_ns
                    .saturating_add(i64::try_from(row.valid.as_nanos()).unwrap_or(i64::MAX)),
                actions,
                tick: 0,
            };
            let token = self
                .leases
                .group()
                .lease(part.channel, self.participant.clock().now_ns())
                .map(|lease| lease.token);
            let publishers = self.leases.publishers();
            let outcome = match publishers
                .get(&part.channel)
                .filter(|p| Some(p.token()) == token)
            {
                Some(publisher) => match publisher.send(draft) {
                    Ok(seq) => Sent::Seq(seq),
                    // Its key went with a token that just ended, before
                    // the lease group saw it: no lease.
                    Err(SessionError::Key(_)) => Sent::NoLease,
                    Err(err) => return Err(err.into()),
                },
                None => Sent::NoLease,
            };
            sent.push((row.entry.clone(), outcome));
        }
        Ok(sent)
    }

    /// Runs the policy loop until [`Self::stop`], `duration`, or a
    /// failure, then gives the leases back. Blocks the calling thread,
    /// which is where `predict` runs.
    pub fn run<P: Predictor>(
        &self,
        predictor: &mut P,
        duration: Option<Duration>,
    ) -> Result<(), RunError<P::Error>> {
        let run = self.runs.fetch_add(1, Ordering::Relaxed) + 1;
        self.stop.store(false, Ordering::Relaxed);
        let clock = self.participant.clock().clone();
        let now = || clock.now_ns();
        let end =
            duration.map(|d| now().saturating_add(i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)));
        let mut recovered_at: Option<i64> = None;
        let mut clock_refused = Vec::new();
        let mut held = Vec::new();
        let mut waiting = None;
        // Wall clock only: virtual and lockstep starts stay on their ticks.
        let wall = self
            .participant
            .authority_view()
            .with(|v| v.is_none_or(|v| v.clock_mode == ClockMode::Wall));
        let mut phase = if wall {
            PhaseAlign::for_binding(&self.binding())
        } else {
            None
        };
        // The schedule runs on `local − offset`: a phase offset delays the
        // whole grid without touching the schedule.
        let offset = |phase: &Option<PhaseAlign>| phase.as_ref().map_or(0, |p| p.offset_ns);
        let result = loop {
            if self.stop.load(Ordering::Relaxed) || self.runs.load(Ordering::Relaxed) != run {
                break Ok(());
            }
            if let Some(reason) = self.lost() {
                // A run this process stopped ends its policies' runs.
                if reason.code == ReasonCode::SceneEnded && self.stopped_here() {
                    break Ok(());
                }
                break Err(RunError::Lost(reason));
            }
            if let Some(why) = self.rebinder.sources().failed.clone() {
                break Err(RunError::Rebind(why));
            }
            self.report_waiting(&mut waiting);
            let off = offset(&phase);
            self.phase_offset_ns.store(off, Ordering::Relaxed);
            let t = now();
            if end.is_some_and(|end| t >= end) {
                break Ok(());
            }
            let next = self.schedule().next_start(t - off) + off;
            if next > t {
                let wake = next.min(t + IDLE_SLICE_NS);
                let wake = end.map_or(wake, |end| wake.min(end));
                sleep_until(now, wake);
                if let Err(err) = predictor.idle() {
                    break Err(RunError::Predict(err));
                }
                continue;
            }
            let (observation, snapshot_ns) = match self.observe() {
                Ok(observed) => observed,
                Err(ObserveError::Skipped(skip)) => {
                    let reason = self.skip_reason(&skip);
                    self.skip(t - off, reason);
                    // A gap inside a window: ask the host's store to fill it
                    // (recovery), at most every 100 ms.
                    if reason == SkipReason::WindowGap
                        && recovered_at.is_none_or(|at| t - at >= RECOVER_GAP_NS)
                    {
                        recovered_at = Some(t);
                        let bound = self.rebinder.bound();
                        match self
                            .runtime
                            .block_on(bound.inputs.recover(&self.participant))
                        {
                            Ok(n) => debug!(samples = n, "window gap recovery"),
                            Err(err) => warn!(%err, "window gap recovery failed"),
                        }
                    }
                    continue;
                }
                Err(err) => {
                    warn!(%err, "start skipped");
                    self.skip(t - off, SkipReason::Decode);
                    continue;
                }
            };
            if let Some(phase) = phase.as_mut() {
                phase.sample(&observation.meta, snapshot_ns);
            }
            // Warm before authority: the first complete snapshot asks for
            // the leases.
            self.want_leases();
            self.report_held(&mut held);
            // Any lease held is enough: a robot that is gone or stopped on
            // its own never starves the others (its actions go unsent).
            let commands = self.binding().actions.iter().any(|a| a.part.is_some());
            if commands && !self.holds_any() {
                self.skip(t - off, SkipReason::NoLease);
                if let Some(phase) = phase.as_mut() {
                    phase.update();
                }
                continue;
            }
            self.schedule().start(now() - off);
            let based_on = observation.meta.based_on.clone();
            let output = predictor.predict(observation, snapshot_ns);
            let finished = self.schedule().finish(now() - off);
            if let Some(phase) = phase.as_mut() {
                phase.update();
            }
            if let Some(event) = finished {
                push_event(&self.events, event);
            }
            let values = match output {
                Ok(values) => values,
                Err(err) => break Err(RunError::Predict(err)),
            };
            match self.send(&values, &based_on, snapshot_ns) {
                Ok(sent) => {
                    self.report_clock_refusals(&mut clock_refused);
                    let mut schedule = self.schedule();
                    for (_, outcome) in sent {
                        if let Sent::Seq(_) = outcome {
                            schedule.sent(true);
                        }
                    }
                }
                Err(err) => {
                    self.schedule().sent(false);
                    break Err(RunError::Send(err));
                }
            }
        };
        self.leases.want.store(false, Ordering::Relaxed);
        if let Err(err) = self
            .runtime
            .block_on(self.leases.release(&self.participant))
        {
            warn!(%err, "releasing the leases after the run failed");
        }
        result
    }

    /// Why the session is gone for this controller: the host, or its
    /// robot's roster entry left absent. In a scene's run a lost robot may
    /// rejoin: only the run closing ends it.
    pub fn lost(&self) -> Option<Reason> {
        self.participant.host_lost().or_else(|| {
            self.participant.authority_view().with(|view| {
                let view = view?;
                if let Some(scene) = &view.scene {
                    return scene.closing.clone();
                }
                let robot = view.participant(&self.binding().robot)?;
                (!robot.present).then(|| robot.left.clone().unwrap_or_default())
            })
        })
    }

    /// Whether this process stopped the scene's run this controller is in.
    #[cfg(not(feature = "fleet"))]
    #[allow(clippy::unused_self)]
    fn stopped_here(&self) -> bool {
        false
    }

    /// Whether this process stopped the scene's run this controller is in.
    #[cfg(feature = "fleet")]
    fn stopped_here(&self) -> bool {
        self.participant.authority_view().with(|view| {
            view.and_then(|v| v.scene.as_ref())
                .is_some_and(|scene| crate::lobby::stopped_here(&scene.run))
        })
    }

    fn schedule(&self) -> MutexGuard<'_, StartSchedule> {
        self.schedule
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Why a start is skipped: `publisher_gone` when an input it waits for
    /// or finds stale has no publisher any more.
    fn skip_reason(&self, skip: &crate::cache::Skip) -> SkipReason {
        let reason = skip.reason();
        if reason == SkipReason::WindowGap {
            return reason;
        }
        let sources = self.rebinder.sources();
        let gone = |entry: &String| sources.gone.iter().any(|g| &g.entry == entry);
        if skip.waiting.iter().chain(&skip.stale).any(gone) {
            SkipReason::PublisherGone
        } else {
            reason
        }
    }

    fn skip(&self, now_ns: i64, reason: SkipReason) {
        let event = self.schedule().skip(now_ns, reason);
        if let Some(event) = event {
            push_event(&self.events, event);
        }
    }

    /// Asks a [`Self::run`] in progress to return.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// How far the run loop's starts sit after the replan grid's first
    /// slot, aligned to an input's publication (0 until aligned, and on
    /// the virtual and lockstep clocks).
    pub fn phase_offset(&self) -> Duration {
        Duration::from_nanos(self.phase_offset_ns.load(Ordering::Relaxed).unsigned_abs())
    }

    /// The input the run loop's starts are aligned to, if any.
    pub fn aligned_input(&self) -> Option<String> {
        let binding = self.binding();
        PhaseAlign::input(&binding).map(|i| binding.inputs[i].entry.clone())
    }

    /// The start and completion counters.
    pub fn stats(&self) -> RunStats {
        self.schedule().stats()
    }

    /// Takes the events reported so far (`rate_drift`, inference pending),
    /// oldest first.
    pub fn drain_events(&self) -> Vec<RunEvent> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect()
    }
}

impl Drop for Controller {
    fn drop(&mut self) {
        self.participant.set_run_source(None);
        self.cancel.cancel();
    }
}

fn action_names(binding: &Binding) -> String {
    let names: Vec<String> = binding
        .actions
        .iter()
        .map(|a| format!("{:?}", a.entry))
        .collect();
    if names.is_empty() {
        "none".into()
    } else {
        names.join(", ")
    }
}

/// Sleeps until `deadline_ns` on `now`'s clock: a coarse sleep, then a
/// short spin, so a 1 kHz loop keeps its period.
pub fn sleep_until(now: impl Fn() -> i64, deadline_ns: i64) {
    loop {
        let left = deadline_ns - now();
        if left <= 0 {
            return;
        }
        if left > 200_000 {
            std::thread::sleep(Duration::from_nanos((left - 150_000).unsigned_abs()));
        } else {
            std::hint::spin_loop();
        }
    }
}
