//! The controller runtime: a bound policy's inputs, leases, chunks and run
//! loop (§6.5, §8.8-§8.10, §9.7). Language bindings wrap it; users write
//! `predict` and nothing else.
//!
//! - [`Controller::bind`] waits for the robot's parts, binds the policy
//!   (refuse-and-explain, §6.7), declares the binding and opens its inputs
//!   ([`Inputs`]).
//! - **Leases.** A background keeper holds the binding's leases while they
//!   are wanted, as one [`LeaseGroup`] (the holder state machine the
//!   browser binding runs too, with its local deadline): it requests them, re-requests them after a preemption
//!   ends, a lapse or a re-arm (each with a fresh token), and declares a
//!   chunk sender per granted lease. On the wall clock it waits until the
//!   host has mapped this controller's clock and its robots' (an edge
//!   rejects chunks until then). [`Controller::acquire`] takes them at
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
//!   leases; the edge holds, then goes safe. If its robot is lost
//!   (`ROBOT_LOST`) or the host is (`HOST_LOST`), the run stops with
//!   [`RunError::Lost`].
//!
//! [`Controller::run`] and [`Controller::send`] block: call them from a
//! plain thread, not from inside the async runtime.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use crate::spec::apply::{ApplyError, Value};
use crate::spec::policy::ChunkStart;
use crate::spec::{BindError, Binding, Policy, RunOptions};
use crate::wire::{
    ChannelId, ClockMode, InputRange, Lease, Mode, ParticipantId, Reason, RunCounters,
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
    /// The session is gone for this run: its robot's heartbeats lapsed
    /// (`ROBOT_LOST`) or the host's liveliness went away (`HOST_LOST`).
    #[error("the session was lost: {0}")]
    Lost(Reason),
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
                info!(participant = %participant.id(), %channel, %reason, "lease lost");
                self.publishers().remove(&channel);
            }
        }
    }

    fn holds_all(&self, participant: &Participant) -> bool {
        self.group().holds_all(participant.clock().now_ns())
    }

    /// Whether the host has mapped this controller's clock and the clock of
    /// every robot it commands. An edge rejects a chunk until both are
    /// mapped (`NoClockMapping`), so leases taken earlier only produce
    /// rejected chunks; a ticked session needs no mapping.
    fn clocks_mapped(&self, participant: &Participant) -> bool {
        let me = participant.id();
        let channels = self.group().channels();
        participant.authority_view().with(|view| {
            let Some(view) = view else {
                return false;
            };
            if view.clock_mode != ClockMode::Wall {
                return true;
            }
            view.clock(me).is_some()
                && channels.iter().all(|c| {
                    view.part(*c)
                        .is_some_and(|part| view.clock(&part.owner).is_some())
                })
        })
    }

    /// Requests every lease, all or none, and declares a sender per new
    /// token.
    async fn grant(&self, participant: &Participant) -> Result<Vec<Lease>, SessionError> {
        let _request = self.request.lock().await;
        let ttl_ns = self
            .ttl
            .map(|ttl| u64::try_from(ttl.as_nanos()).unwrap_or(u64::MAX));
        let command = self.group().request(
            participant.binding_fingerprint(),
            ttl_ns,
            participant.clock().now_ns(),
        );
        let reply = participant.command(command).await?;
        self.group().on_reply(
            &reply,
            participant.clock_estimate().as_ref(),
            participant.clock().now_ns(),
        );
        reply.result.clone()?;
        for lease in &reply.leases {
            let current = self
                .publishers()
                .get(&lease.channel)
                .is_some_and(|p| p.token() == lease.token);
            if current {
                continue;
            }
            let publisher = participant.chunk_publisher(lease).await?;
            info!(participant = %participant.id(), channel = %lease.channel, token = %lease.token, "lease granted");
            self.publishers().insert(lease.channel, publisher);
        }
        Ok(reply.leases)
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
        self.leases.sync(&self.participant);
        self.leases.holds_all(&self.participant)
    }

    /// The leases held now.
    pub fn leases(&self) -> Vec<Lease> {
        self.leases.sync(&self.participant);
        self.leases.group().held(self.participant.clock().now_ns())
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
            if let Some(reason) = self.lost() {
                break Err(RunError::Lost(reason));
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
                    let reason = skip.reason();
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
                phase.sample(&observation.meta, snapshot_ns);
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

    /// Why the session is gone for this controller: the host, or its
    /// robot's roster entry left absent.
    pub fn lost(&self) -> Option<Reason> {
        self.participant.host_lost().or_else(|| {
            self.participant.authority_view().with(|view| {
                let robot = view?.participant(&self.binding.robot)?;
                (!robot.present).then(|| robot.left.clone().unwrap_or_default())
            })
        })
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
        self.participant.set_run_source(None);
        self.cancel.cancel();
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
