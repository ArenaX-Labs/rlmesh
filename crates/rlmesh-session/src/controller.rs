//! The controller runtime: a bound policy's inputs, leases, chunks and run
//! loop (§6.5, §8.8-§8.10, §9.7). Language bindings wrap it; users write
//! `predict` and nothing else.
//!
//! - [`Controller::bind`] waits for the robot's parts, binds the policy
//!   (refuse-and-explain, §6.7), declares the binding and opens its inputs
//!   ([`Inputs`]).
//! - **Leases.** A background keeper holds the binding's leases while they
//!   are wanted: it requests them, re-requests them after a preemption
//!   ends, a lapse or a re-arm (each with a fresh token), and declares a
//!   chunk sender per granted lease. [`Controller::acquire`] takes them at
//!   once (an operator taking over), [`Controller::release`] gives them
//!   back.
//! - [`Controller::send`] packs `predict`'s output per action entry
//!   (`ActionRow::pack`) and sends one timed chunk per commanded part:
//!   `t0` at the snapshot (or at send, `t0: "send"`), the binding's action
//!   grid, `valid_until = t0 + min(horizon × dt, ttl)`, `replace_from = t0`
//!   and the snapshot's `based_on`.
//! - [`Controller::run`] is the policy loop of §8.9: starts on the replan
//!   grid, one inference in flight, skipped starts with their reasons, warm
//!   before authority (leases are requested only once a first snapshot of
//!   the required inputs is complete), window gaps recovered from the
//!   host's history store, and §8.10 reporting through [`StartSchedule`].
//!   On the wall clock the grid is phased just after the publication of
//!   one sampled input ([`PhaseAlign`]), so a snapshot reads a sample about
//!   a millisecond old rather than up to a full replan period old.
//!   If `predict` fails, the controller stops sending and releases its
//!   leases; the edge holds, then goes safe.
//!
//! [`Controller::run`] and [`Controller::send`] block: call them from a
//! plain thread, not from inside the async runtime.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use bytes::Bytes;
use rlmesh_spec::apply::{ApplyError, Value};
use rlmesh_spec::policy::ChunkStart;
use rlmesh_spec::{BindError, Binding, Policy, RunOptions};
use rlmesh_wire::{
    ChannelId, ClockMode, ContractMode, Epoch, InputRange, Lease, Mode, ParticipantId,
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::cache::{Inputs, Observation, ObserveError, Skip};
use crate::chunk::{ChunkDraft, ChunkPublisher};
use crate::error::SessionError;
use crate::participant::Participant;
use crate::schedule::{RunEvent, RunStats, SkipReason, StartSchedule};

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
    /// How long [`Controller::bind`] waits for the robot's parts.
    pub bind_timeout: Duration,
    /// How often a wanted lease is re-requested while it is refused (it is
    /// also re-requested on every authority change).
    pub lease_retry: Duration,
    /// An inference running this long is reported (§8.10).
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
    /// The binder refused (every problem, §6.7) or the options do not fit.
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
}

/// The model side of [`Controller::run`].
pub trait Predictor {
    /// The model's error.
    type Error;

    /// One start: the observation (snapshotted at `snapshot_ns`, session
    /// time) in, one value per action entry out (`[horizon, ...]`, §6.1).
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

struct Held {
    lease: Lease,
    epoch: Epoch,
    publisher: ChunkPublisher,
}

/// The lease keeper's shared state.
struct Leases {
    channels: Vec<ChannelId>,
    want: AtomicBool,
    held: Mutex<BTreeMap<ChannelId, Held>>,
    /// Serialises grants and releases between the keeper and explicit
    /// calls.
    request: tokio::sync::Mutex<()>,
    wake: Notify,
    ttl: Option<Duration>,
}

impl Leases {
    fn held(&self) -> MutexGuard<'_, BTreeMap<ChannelId, Held>> {
        self.held
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Drops chunk senders the authority view no longer backs (lost to a
    /// preemption, lapsed, or voided by a new epoch).
    fn prune(&self, participant: &Participant) {
        let me = participant.id();
        participant.authority_view().with(|view| {
            let Some(view) = view else {
                return;
            };
            self.held().retain(|channel, held| {
                let kept = view.epoch == held.epoch
                    && view
                        .lease(*channel)
                        .is_some_and(|l| &l.holder == me && l.token == held.lease.token);
                if !kept {
                    info!(participant = %me, %channel, token = %held.lease.token, "lease lost");
                }
                kept
            });
        });
    }

    fn holds_all(&self) -> bool {
        let held = self.held();
        self.channels.iter().all(|c| held.contains_key(c))
    }

    /// Requests every lease, all or none, and declares a sender per new
    /// token.
    async fn grant(&self, participant: &Participant) -> Result<Vec<Lease>, SessionError> {
        let _request = self.request.lock().await;
        let leases = participant
            .grant_leases(self.channels.clone(), Vec::new(), self.ttl)
            .await?;
        let epoch = participant.authority_view().epoch();
        for lease in &leases {
            let current = self
                .held()
                .get(&lease.channel)
                .is_some_and(|h| h.lease.token == lease.token && h.epoch == epoch);
            if current {
                continue;
            }
            let publisher = participant.chunk_publisher(lease).await?;
            info!(participant = %participant.id(), channel = %lease.channel, token = %lease.token, "lease granted");
            self.held().insert(
                lease.channel,
                Held {
                    lease: lease.clone(),
                    epoch,
                    publisher,
                },
            );
        }
        Ok(leases)
    }

    /// Gives every held lease back.
    async fn release(&self, participant: &Participant) -> Result<(), SessionError> {
        let _request = self.request.lock().await;
        let by_token: BTreeMap<_, Vec<ChannelId>> =
            self.held()
                .values()
                .fold(BTreeMap::new(), |mut by_token, held| {
                    by_token
                        .entry(held.lease.token)
                        .or_insert_with(Vec::new)
                        .push(held.lease.channel);
                    by_token
                });
        self.held().clear();
        for (token, channels) in by_token {
            match participant.release_leases(channels, token).await {
                // Already gone (preempted, lapsed, a new epoch): nothing to
                // give back.
                Ok(_) | Err(SessionError::Rejected(_)) => {}
                Err(err) => return Err(err),
            }
        }
        Ok(())
    }
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
        leases.prune(&participant);
        let wanted = leases.want.load(Ordering::Relaxed);
        let mode = participant.authority().map(|v| v.mode);
        if wanted && !leases.holds_all() && matches!(mode, Some(Mode::Armed | Mode::Active)) {
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

/// Reports an inference pending past the configured bound (§8.10: a timer,
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
    binding: Binding,
    inputs: Inputs,
    leases: Arc<Leases>,
    schedule: Arc<Mutex<StartSchedule>>,
    events: Arc<Mutex<VecDeque<RunEvent>>>,
    runtime: tokio::runtime::Handle,
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
    /// binding and opens its inputs. Call it on a Tokio runtime: the
    /// controller's background tasks run there.
    pub async fn bind(
        participant: Arc<Participant>,
        policy: &Policy,
        options: &RunOptions,
        config: ControllerConfig,
    ) -> Result<Self, ControllerError> {
        let robot: Option<ParticipantId> = options.robot.clone();
        participant
            .wait_for("the robot's parts", config.bind_timeout, |view| {
                view.parts
                    .iter()
                    .any(|p| robot.as_ref().is_none_or(|r| &p.owner == r))
            })
            .await?;
        let binding = participant.bind(policy, options)?;
        participant.declare_bound(&binding).await?;
        let inputs = Inputs::open(&participant, &binding).await?;
        let leases = Arc::new(Leases {
            channels: binding.channels(),
            want: AtomicBool::new(false),
            held: Mutex::new(BTreeMap::new()),
            request: tokio::sync::Mutex::new(()),
            wake: Notify::new(),
            ttl: config.lease_ttl,
        });
        let schedule = Arc::new(Mutex::new(StartSchedule::new(binding.replan_hz)));
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
        ];
        Ok(Self {
            participant,
            binding,
            inputs,
            leases,
            schedule,
            events,
            runtime: tokio::runtime::Handle::current(),
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

    /// The binding table.
    pub fn binding(&self) -> &Binding {
        &self.binding
    }

    /// The participant.
    pub fn participant(&self) -> &Arc<Participant> {
        &self.participant
    }

    /// A policy start now: the snapshot (session time), each input through
    /// its plan, and the session time it was taken at.
    pub fn observe(&self) -> Result<(Observation, i64), ObserveError> {
        let now = self.participant.session_now_ns();
        self.inputs.observe(now).map(|o| (o, now))
    }

    /// Whether this controller holds every lease its binding commands.
    pub fn holds(&self) -> bool {
        self.leases.prune(&self.participant);
        self.leases.holds_all()
    }

    /// The leases held now.
    pub fn leases(&self) -> Vec<Lease> {
        self.leases.prune(&self.participant);
        self.leases
            .held()
            .values()
            .map(|h| h.lease.clone())
            .collect()
    }

    /// Takes the binding's leases now (all or none), preempting a lower
    /// class; from then on the keeper holds them. A refusal (`LEASE_BUSY`,
    /// `GROUP_BUSY`, `NOT_READY`, ...) leaves the keeper as it was.
    pub async fn acquire(&self) -> Result<Vec<Lease>, SessionError> {
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
        if let Some((entry, _)) = values
            .iter()
            .find(|(entry, _)| self.binding.action(entry).is_none())
        {
            return Err(SendError::Value(format!(
                "{entry:?} is not an action of this policy (its actions: {})",
                self.action_names()
            )));
        }
        let mut sent = Vec::with_capacity(self.binding.actions.len());
        for row in &self.binding.actions {
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
            let held = self.leases.held();
            let outcome = match held.get(&part.channel) {
                Some(held) => Sent::Seq(held.publisher.send(draft)?),
                None => Sent::NoLease,
            };
            sent.push((row.entry.clone(), outcome));
        }
        Ok(sent)
    }

    fn action_names(&self) -> String {
        let names: Vec<String> = self
            .binding
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

    /// Runs the policy loop (§8.9) until [`Self::stop`], `duration`, or a
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
        let commands = !self.binding.channels().is_empty();
        let mut recovered_at: Option<i64> = None;
        // Wall clock only: virtual and lockstep starts stay on their ticks.
        let wall = self
            .participant
            .authority_view()
            .with(|v| v.is_none_or(|v| v.clock_mode == ClockMode::Wall));
        let mut phase = if wall {
            PhaseAlign::for_binding(&self.binding)
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
                    let reason = skip_reason(&skip);
                    self.skip(t - off, reason);
                    // A gap inside a window: ask the host's store to fill it
                    // (§8.8 recovery), at most every 100 ms.
                    if reason == SkipReason::WindowGap
                        && recovered_at.is_none_or(|at| t - at >= RECOVER_GAP_NS)
                    {
                        recovered_at = Some(t);
                        match self
                            .runtime
                            .block_on(self.inputs.recover(&self.participant))
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
                phase.sample(&observation, snapshot_ns);
            }
            // Warm before authority: the first complete snapshot asks for
            // the leases.
            self.want_leases();
            if commands && !self.holds() {
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

    fn schedule(&self) -> MutexGuard<'_, StartSchedule> {
        self.schedule
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
    pub fn aligned_input(&self) -> Option<&str> {
        PhaseAlign::input(&self.binding).map(|i| self.binding.inputs[i].entry.as_str())
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
        self.cancel.cancel();
    }
}

/// Phases a clocked run loop's starts just after the publication of one
/// sampled input (wall clock only; sans-IO, times injected).
///
/// A `latest` input is published at about the replan rate, at a phase set
/// by its publisher, and the start grid's phase is set by when the run
/// began: the two lock wherever they happen to fall, often with each start
/// just before the next sample, so every snapshot reads a sample nearly a
/// full replan period old. The aligner measures the phase of the aligned
/// input's newest sample at each snapshot (`(snapshot − capture) mod P`),
/// takes the freshest tenth over a window of starts and moves the
/// grid so that phase is [`Self::margin`]: once in full (always later, so no slot
/// is skipped), then by half the remaining error per window, at most a
/// quarter period, to follow slow drift between the publisher's clock and
/// this one.
///
/// The aligned input is the first bound input with a `latest` contract
/// (its publication follows the replan rate). `every` and `history`
/// inputs are already fresh at any phase, and events have none.
#[derive(Clone, Debug)]
pub struct PhaseAlign {
    input: usize,
    period_ns: i64,
    /// How far starts sit after the schedule's grid.
    pub offset_ns: i64,
    errors: Vec<i64>,
    aligned: bool,
}

impl PhaseAlign {
    /// Starts per estimate.
    pub const WINDOW: usize = 32;
    /// Starts before the first estimate.
    pub const FIRST: usize = 8;
    /// Consecutive starts reading the previous sample that move the
    /// starts later before the window ends.
    pub const LATE: usize = 4;
    /// The target sample age at a start: transit plus the publisher's
    /// loop jitter (`min(1 ms, P / 4)`).
    const MARGIN_NS: i64 = 1_000_000;
    /// Corrections smaller than this are left alone.
    const DEADBAND_NS: i64 = 50_000;

    /// The input `binding`'s starts align to, by index.
    pub fn input(binding: &Binding) -> Option<usize> {
        binding.inputs.iter().position(|row| {
            row.channel.is_some_and(|channel| {
                binding
                    .contracts
                    .iter()
                    .any(|c| c.channel == channel && c.mode == ContractMode::Latest)
            })
        })
    }

    /// An aligner for `binding`, or `None` when no input sets a phase.
    pub fn for_binding(binding: &Binding) -> Option<Self> {
        let input = Self::input(binding)?;
        let hz = if binding.replan_hz.is_finite() && binding.replan_hz > 0.0 {
            binding.replan_hz
        } else {
            return None;
        };
        Some(Self::new(input, ((1e9 / hz).round() as i64).max(1)))
    }

    /// Aligns to input `input` on a grid of period `period_ns`.
    pub fn new(input: usize, period_ns: i64) -> Self {
        Self {
            input,
            period_ns,
            offset_ns: 0,
            errors: Vec::with_capacity(Self::WINDOW),
            aligned: false,
        }
    }

    /// The target phase.
    pub fn margin(&self) -> i64 {
        Self::MARGIN_NS.min(self.period_ns / 4)
    }

    /// Records the phase of the aligned input's newest sample at a snapshot
    /// taken at `snapshot_ns` (session time).
    pub fn sample(&mut self, observation: &Observation, snapshot_ns: i64) {
        let capture = observation
            .meta
            .inputs
            .get(self.input)
            .and_then(|i| i.samples.last())
            .and_then(|s| s.capture_ns);
        if let Some(capture) = capture {
            self.record(snapshot_ns - capture);
        }
    }

    /// Records one sample age (snapshot minus capture).
    pub fn record(&mut self, age_ns: i64) {
        let phase = age_ns.rem_euclid(self.period_ns);
        self.errors.push(self.centered(phase - self.margin()));
    }

    /// Moves the grid once a window is full; returns the change.
    pub fn update(&mut self) -> i64 {
        // Every one of the last few starts read the previous sample: the
        // publisher's phase moved later (its loop stalled). Follow now
        // rather than at the end of the window, by the smallest of them.
        let recent = self.errors.len().checked_sub(Self::LATE);
        if let Some(from) = recent.filter(|_| self.aligned)
            && self.errors[from..]
                .iter()
                .all(|e| (-self.period_ns / 4..-Self::DEADBAND_NS).contains(e))
        {
            let change = self.errors[from..].iter().map(|e| -e).min().unwrap_or(0);
            self.errors.clear();
            self.offset_ns += change;
            return change;
        }
        let window = if self.aligned {
            Self::WINDOW
        } else {
            Self::FIRST
        };
        if self.errors.len() < window {
            return 0;
        }
        self.errors.sort_unstable();
        // The freshest tenth: the samples that land latest set the phase,
        // so jitter in their arrival never puts one after the start.
        let error = self.errors[self.errors.len() / 10];
        self.errors.clear();
        // A phase above the margin: start that much earlier, which is the
        // same as starting `P − error` later on the next sample.
        let p = self.period_ns;
        let change = if (-p / 4..0).contains(&error) {
            // The samples land just after the starts: start later, in full.
            // Each start reads a sample almost a period old until then, and
            // the gap this opens in the chunk coverage is under P/4.
            -error
        } else if error < 0 {
            // Far later is the same phase as earlier: start sooner (which
            // never opens a gap), at most P/2 per window.
            (-(error + p)).max(-p / 2)
        } else if self.aligned {
            // Starts later than they need be: close in by halves.
            (-error / 2).max(-p / 4)
        } else {
            -error
        };
        self.aligned = true;
        if change.abs() < Self::DEADBAND_NS {
            return 0;
        }
        self.offset_ns += change;
        change
    }

    /// `x` wrapped into `(−P/2, P/2]`.
    fn centered(&self, x: i64) -> i64 {
        let p = self.period_ns;
        let w = x.rem_euclid(p);
        if w > p / 2 { w - p } else { w }
    }
}

/// The reason a snapshot was refused, most specific first.
fn skip_reason(skip: &Skip) -> SkipReason {
    if !skip.gaps.is_empty() {
        SkipReason::WindowGap
    } else if !skip.stale.is_empty() {
        SkipReason::StaleInput
    } else {
        SkipReason::Waiting
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

#[cfg(test)]
mod tests {
    use super::PhaseAlign;

    const MS: i64 = 1_000_000;

    /// Starts on a 10 ms grid shifted by `offset`; samples published every
    /// 10 ms at `publish_phase` (+ `jitter(k)`): the newest sample's age at
    /// each start.
    fn ages(
        offset: i64,
        publish_phase: i64,
        jitter: impl Fn(i64) -> i64,
        from: i64,
    ) -> impl Iterator<Item = i64> {
        (from..).map(move |k| {
            let start = k * 10 * MS + offset;
            // The newest publication at or before the start.
            let mut j = (start - publish_phase).div_euclid(10 * MS) + 1;
            while j * 10 * MS + publish_phase + jitter(j) > start {
                j -= 1;
            }
            start - (j * 10 * MS + publish_phase + jitter(j))
        })
    }

    #[test]
    fn a_start_just_before_each_sample_moves_to_just_after_it() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        // Published 0.4 ms after each start: snapshots read a 9.6 ms old sample.
        for age in ages(0, 400_000, |_| 0, 1).take(PhaseAlign::FIRST) {
            assert_eq!(age, 9_600_000);
            align.record(age);
        }
        let change = align.update();
        assert_eq!(change, 1_400_000, "later, the short way");
        for age in ages(align.offset_ns, 400_000, |_| 0, 100).take(PhaseAlign::WINDOW) {
            assert_eq!(age, MS, "one margin old");
            align.record(age);
        }
        assert_eq!(align.update(), 0, "aligned: nothing to correct");
    }

    #[test]
    fn jitter_is_ignored_and_slow_drift_is_followed() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        // ±0.5 ms publication jitter around a 3 ms phase.
        let jitter = |j: i64| if j % 2 == 0 { 500_000 } else { -500_000 };
        for age in ages(0, 3 * MS, jitter, 1).take(PhaseAlign::WINDOW) {
            align.record(age);
        }
        align.update();
        let mut worst = 0;
        // The publisher drifts 20 µs per window; the aligner keeps up.
        for round in 0..50 {
            let phase = 3 * MS + round * 20_000;
            let from = 1000 + round * 100;
            for age in ages(align.offset_ns, phase, jitter, from).take(PhaseAlign::WINDOW) {
                worst = worst.max(age);
                align.record(age);
            }
            align.update();
        }
        assert!(worst < 3 * MS, "ages stay near the margin, worst {worst}");
    }

    #[test]
    fn nothing_moves_before_a_window_is_full() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        for _ in 0..PhaseAlign::FIRST - 1 {
            align.record(9 * MS);
        }
        assert_eq!(align.update(), 0);
        assert_eq!(align.offset_ns, 0);
    }

    #[test]
    fn a_far_phase_is_reached_by_starting_earlier() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        // 4 ms old: 3 ms earlier, not 7 ms later.
        for _ in 0..PhaseAlign::FIRST {
            align.record(4 * MS);
        }
        assert_eq!(align.update(), -3 * MS);
        // 6.5 ms after the margin is 3.5 ms before the next sample's: still
        // earlier, by at most half a period per window.
        let mut align = PhaseAlign::new(0, 10 * MS);
        for _ in 0..PhaseAlign::FIRST {
            align.record(7_500_000);
        }
        assert_eq!(align.update(), -5 * MS);
    }

    #[test]
    fn a_publisher_that_jumps_later_is_followed_within_a_few_starts() {
        let mut align = PhaseAlign::new(0, 10 * MS);
        for age in ages(0, 8 * MS, |_| 0, 1).take(PhaseAlign::FIRST) {
            align.record(age);
        }
        assert_eq!(align.update(), -MS, "2 ms old: 1 ms earlier");
        // The publisher's loop slips 2 ms: every start now reads the
        // previous sample, 9 ms old, until the aligner follows.
        let mut moved = 0;
        for (n, age) in ages(align.offset_ns, 10 * MS, |_| 0, 100)
            .take(PhaseAlign::LATE)
            .enumerate()
        {
            assert_eq!(age, 9 * MS);
            align.record(age);
            moved = align.update();
            if n + 1 < PhaseAlign::LATE {
                assert_eq!(moved, 0);
            }
        }
        assert_eq!(moved, 2 * MS, "later, in full, after a few starts");
        for age in ages(align.offset_ns, 10 * MS, |_| 0, 200).take(PhaseAlign::WINDOW) {
            assert_eq!(age, MS);
        }
    }
}
