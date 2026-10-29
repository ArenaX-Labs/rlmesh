//! The cockpit's state, sans-IO: [`Model::ingest`] takes what the observer
//! received ([`Input`]) at an injected time, and [`Model::view`]
//! (in [`crate::view`]) derives everything the UI shows. No clocks are read,
//! nothing sleeps or spawns, and a decode failure is a value, never a panic.
//!
//! Times are this cockpit's clock (`now_ns`, receipt times) unless a name
//! says otherwise: `capture_ns` and `t1_ns` are the publisher's clock,
//! `sent_ns` the reporter's, and `*_session_ns` session time through the
//! host's clock mappings.

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use rlmesh_session::wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, EdgeMode, Heartbeat, PartSpec, ParticipantId,
    ParticipantReport, PublisherCounters, RunCounters, Sample, StreamSpec,
};

use crate::meter::{CounterRate, RateMeter, window_for};
use crate::timeline::{Entry, Kind, Timeline, Tone};
use crate::values::{self, Value};

/// A verb the cockpit sends through the host (§11.1). A latched e-stop is
/// cleared only by the embodiment (a local clear), so there is no clear
/// here: after it, disarm and arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Verb {
    /// DISARMED or SAFE to ARMED, in a fresh epoch.
    Arm,
    /// Back to DISARMED, revoking every lease.
    Disarm,
    /// Emergency stop: latches until the robot clears it.
    Estop,
}

impl Verb {
    /// Its name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Arm => "arm",
            Self::Disarm => "disarm",
            Self::Estop => "e-stop",
        }
    }
}

/// Something the observer received.
#[derive(Clone, Debug)]
pub enum Input {
    /// A newer authority snapshot.
    Authority(Box<AuthoritySnapshot>),
    /// A sample of a part's stream or of an `_applied` stream.
    Sample {
        /// The sample.
        sample: Sample,
        /// Cockpit clock at receipt.
        receive_ns: i64,
        /// Its capture time converted to the cockpit's clock.
        capture_local_ns: i64,
    },
    /// An action chunk on its way to an edge.
    Chunk {
        /// The chunk.
        chunk: ActionChunk,
        /// Cockpit clock at receipt.
        receive_ns: i64,
    },
    /// A participant's heartbeat.
    Heartbeat {
        /// The heartbeat.
        heartbeat: Heartbeat,
        /// Cockpit clock at receipt.
        receive_ns: i64,
    },
    /// A participant's observer report.
    Report {
        /// The report.
        report: ParticipantReport,
        /// Cockpit clock at receipt.
        receive_ns: i64,
    },
    /// The host answered a verb this cockpit sent.
    Outcome {
        /// The verb.
        verb: Verb,
        /// `Ok`, or the refusal.
        result: Result<(), String>,
    },
    /// The host's liveliness token appeared or went away.
    HostPresent(bool),
    /// The cockpit's own input queue has dropped this many inputs in total
    /// (the UI fell behind): shown, so it is never read as network loss.
    Overrun(u64),
}

/// Windows and budgets.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    /// How much of each plotted stream is kept.
    pub plot_window: Duration,
    /// The shortest interval between plotted (and decoded) values.
    pub plot_every: Duration,
    /// The shortest interval between decoded camera frames.
    pub thumbnail_every: Duration,
    /// The rate window of reported counters.
    pub counter_window: Duration,
    /// How far back deadline misses and chunk ages are summarised.
    pub recent: Duration,
    /// How far back the lanes reach.
    pub lane_window: Duration,
    /// How many timeline entries are kept.
    pub timeline_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            plot_window: Duration::from_secs(10),
            plot_every: Duration::from_millis(20),
            thumbnail_every: Duration::from_millis(200),
            counter_window: Duration::from_secs(2),
            recent: Duration::from_secs(10),
            lane_window: Duration::from_secs(60),
            timeline_len: 512,
        }
    }
}

pub(crate) fn ns(d: Duration) -> i64 {
    i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)
}

/// One stream as received.
#[derive(Clone, Debug)]
pub(crate) struct StreamState {
    pub(crate) spec: StreamSpec,
    pub(crate) names: Vec<String>,
    pub(crate) image: bool,
    /// Unique samples per second of the cockpit's clock.
    pub(crate) received: RateMeter,
    /// Samples per second of capture time (the publisher's clock: session
    /// time in virtual and lockstep modes).
    pub(crate) captured: RateMeter,
    /// `(seq, capture_local_ns, receive_ns)` of the newest.
    pub(crate) last: Option<(u64, i64, i64)>,
    /// The newest raw capture stamp (the publisher's clock).
    pub(crate) captured_newest: i64,
    /// Seqs skipped (losses between publisher and cockpit).
    pub(crate) lost: u64,
    pub(crate) value: Option<Value>,
    pub(crate) decoded_ns: i64,
    /// `(receive_ns, first values)`, oldest first.
    pub(crate) plot: VecDeque<(i64, Vec<f64>)>,
}

/// A publisher's counters for one stream, from its reports.
#[derive(Clone, Debug)]
pub(crate) struct PublisherState {
    pub(crate) produced: CounterRate,
    pub(crate) published: CounterRate,
    pub(crate) last: PublisherCounters,
}

/// One edge (one commandable part), from its `_applied` stream.
#[derive(Clone, Debug)]
pub(crate) struct EdgeState {
    pub(crate) period_ns: i64,
    /// Ticks per second of the edge's own tick stamps.
    pub(crate) ticks: RateMeter,
    /// `(seq, capture_ns)` of the newest tick, on the edge's clock.
    pub(crate) last: Option<(u64, i64)>,
    /// Cockpit clock when the newest tick arrived.
    pub(crate) last_receive_ns: i64,
    /// Ticks later than two control periods (§9.1).
    pub(crate) misses: u64,
    /// Cockpit clock of each recent miss.
    pub(crate) recent_misses: VecDeque<i64>,
    /// The longest gap between consecutive ticks.
    pub(crate) longest_gap_ns: i64,
    pub(crate) mode: Option<(EdgeMode, i64)>,
    pub(crate) present: u64,
    pub(crate) total: u64,
}

/// The newest chunk sent to one commandable part.
#[derive(Clone, Debug)]
pub(crate) struct LastChunk {
    pub(crate) source: ParticipantId,
    pub(crate) token: rlmesh_session::wire::LeaseToken,
    pub(crate) seq: u64,
    pub(crate) t0_ns: i64,
    pub(crate) dt_ns: i64,
    pub(crate) count: u32,
    pub(crate) valid_until_ns: i64,
    pub(crate) receipt_session_ns: i64,
}

/// Chunks sent to one commandable part.
#[derive(Clone, Debug)]
pub(crate) struct ChunkState {
    pub(crate) received: RateMeter,
    pub(crate) last: Option<LastChunk>,
    /// Chunks under a token or epoch that is no longer current.
    pub(crate) fenced: u64,
}

/// One controller runtime.
#[derive(Clone, Debug)]
pub(crate) struct ControllerState {
    pub(crate) starts: CounterRate,
    pub(crate) completions: CounterRate,
    pub(crate) busy: CounterRate,
    pub(crate) last: Option<RunCounters>,
    /// Chunks per second it sent, on the cockpit's clock.
    pub(crate) chunks: RateMeter,
    /// `(receive_ns, age at receipt)` of its recent chunks: receipt minus
    /// publish, in session time.
    pub(crate) ages: VecDeque<(i64, i64)>,
}

/// The cockpit's state; see the module docs.
#[derive(Clone, Debug)]
pub struct Model {
    pub(crate) me: ParticipantId,
    pub(crate) config: ModelConfig,
    pub(crate) snapshot: Option<AuthoritySnapshot>,
    pub(crate) host_present: bool,
    pub(crate) streams: BTreeMap<ChannelId, StreamState>,
    pub(crate) publishers: BTreeMap<ChannelId, PublisherState>,
    /// By the commandable part's channel.
    pub(crate) edges: BTreeMap<ChannelId, EdgeState>,
    /// By the commandable part's channel.
    pub(crate) chunks: BTreeMap<ChannelId, ChunkState>,
    pub(crate) controllers: BTreeMap<ParticipantId, ControllerState>,
    /// The newest heartbeat per participant, with its receipt time.
    pub(crate) heartbeats: BTreeMap<ParticipantId, (Heartbeat, i64)>,
    /// How fast each participant's newest seen seq of each stream advances,
    /// on its clock (heartbeat `last_seq_seen`).
    pub(crate) heard: BTreeMap<(ParticipantId, ChannelId), CounterRate>,
    /// `(receive_ns, newest capture_ns)`: session time against the
    /// cockpit's clock, for the real-time factor.
    pub(crate) progress: VecDeque<(i64, i64)>,
    pub(crate) timeline: Timeline,
    pub(crate) last_outcome: Option<(Verb, Result<(), String>, i64)>,
    pub(crate) overrun: u64,
}

impl Model {
    /// An empty model for the observer `me`.
    pub fn new(me: ParticipantId, config: ModelConfig) -> Self {
        Self {
            me,
            timeline: Timeline::new(config.timeline_len),
            config,
            snapshot: None,
            host_present: false,
            streams: BTreeMap::new(),
            publishers: BTreeMap::new(),
            edges: BTreeMap::new(),
            chunks: BTreeMap::new(),
            controllers: BTreeMap::new(),
            heartbeats: BTreeMap::new(),
            heard: BTreeMap::new(),
            progress: VecDeque::new(),
            last_outcome: None,
            overrun: 0,
        }
    }

    /// The observer this cockpit attached as.
    pub fn me(&self) -> &ParticipantId {
        &self.me
    }

    /// The newest authority snapshot.
    pub fn snapshot(&self) -> Option<&AuthoritySnapshot> {
        self.snapshot.as_ref()
    }

    /// Session time at `now_ns` on the cockpit's clock, through its own
    /// clock mapping (unconverted until the host has one).
    pub fn session_ns(&self, now_ns: i64) -> i64 {
        now_ns.saturating_add(self.offset(&self.me))
    }

    pub(crate) fn offset(&self, id: &ParticipantId) -> i64 {
        self.snapshot
            .as_ref()
            .and_then(|s| s.clock(id))
            .map_or(0, |c| c.offset_ns)
    }

    /// Takes in one input received (or happening) at `now_ns`.
    pub fn ingest(&mut self, input: Input, now_ns: i64) {
        match input {
            Input::Authority(snapshot) => self.on_authority(*snapshot, now_ns),
            Input::Sample {
                sample,
                receive_ns,
                capture_local_ns,
            } => self.on_sample(&sample, receive_ns, capture_local_ns),
            Input::Chunk { chunk, receive_ns } => self.on_chunk(&chunk, receive_ns),
            Input::Heartbeat {
                heartbeat,
                receive_ns,
            } => self.on_heartbeat(heartbeat, receive_ns),
            Input::Report { report, receive_ns } => self.on_report(&report, receive_ns),
            Input::Outcome { verb, result } => {
                let (tone, text) = match &result {
                    Ok(()) => (
                        Tone::Neutral,
                        format!("cockpit sent {}: accepted", verb.name()),
                    ),
                    Err(why) => (
                        Tone::Warn,
                        format!("cockpit sent {}: refused: {why}", verb.name()),
                    ),
                };
                self.push(now_ns, Kind::Command, tone, text);
                self.last_outcome = Some((verb, result, now_ns));
            }
            Input::HostPresent(present) => {
                if present != self.host_present {
                    let (tone, text) = if present {
                        (Tone::Neutral, "session host present")
                    } else {
                        (Tone::Bad, "session host lost")
                    };
                    self.push(now_ns, Kind::Roster, tone, text.into());
                }
                self.host_present = present;
            }
            Input::Overrun(total) => self.overrun = self.overrun.max(total),
        }
        self.timeline
            .prune(now_ns.saturating_sub(ns(self.config.lane_window)));
    }

    fn push(&mut self, at_ns: i64, kind: Kind, tone: Tone, text: String) {
        let snapshot = self.snapshot.as_ref();
        self.timeline.push(Entry {
            at_ns,
            session_ns: snapshot.map(|_| self.session_ns(at_ns)),
            epoch: snapshot.map_or(0, |s| s.epoch.0),
            kind,
            tone,
            text,
        });
    }

    fn on_authority(&mut self, snapshot: AuthoritySnapshot, now_ns: i64) {
        if self
            .snapshot
            .as_ref()
            .is_some_and(|old| !snapshot.is_newer_than(old))
        {
            return;
        }
        self.host_present = true;
        self.timeline
            .observe(self.snapshot.as_ref(), &snapshot, now_ns);
        self.snapshot = Some(snapshot);
    }

    /// The part a stream belongs to: its own, or the commandable part whose
    /// `_applied` stream it is.
    pub(crate) fn part_of(&self, channel: ChannelId) -> Option<&PartSpec> {
        self.snapshot
            .as_ref()?
            .parts
            .iter()
            .find(|p| p.channel == channel || p.applied_channel == Some(channel))
    }

    fn on_sample(&mut self, sample: &Sample, receive_ns: i64, capture_local_ns: i64) {
        let config = self.config.clone();
        if !self.streams.contains_key(&sample.channel) {
            let Some(snapshot) = self.snapshot.as_ref() else {
                return;
            };
            let Some(spec) = snapshot.stream(sample.channel) else {
                return;
            };
            let part = self.part_of(sample.channel);
            let names = part.map(|p| p.names.clone()).unwrap_or_default();
            let hz = if spec.applied {
                part.map_or(0.0, |p| p.control_hz)
            } else {
                part.map_or(0.0, |p| p.hz)
            };
            self.streams.insert(
                sample.channel,
                StreamState {
                    image: values::is_image(&spec),
                    spec,
                    names,
                    received: RateMeter::new(window_for(hz)),
                    captured: RateMeter::new(window_for(hz)),
                    last: None,
                    captured_newest: 0,
                    lost: 0,
                    value: None,
                    decoded_ns: i64::MIN,
                    plot: VecDeque::new(),
                },
            );
        }
        let Some(stream) = self.streams.get_mut(&sample.channel) else {
            return;
        };
        if let Some((seq, _, _)) = stream.last {
            if sample.seq <= seq {
                // A replay or a reordered sample: never a new value.
                return;
            }
            stream.lost += sample.seq - seq - 1;
        }
        stream.last = Some((sample.seq, capture_local_ns, receive_ns));
        stream.received.record(receive_ns);
        stream.captured.record(sample.capture_ns);
        stream.captured_newest = stream.captured_newest.max(sample.capture_ns);

        let due_every = if stream.image {
            config.thumbnail_every
        } else {
            config.plot_every
        };
        if receive_ns.saturating_sub(stream.decoded_ns) >= ns(due_every) {
            stream.decoded_ns = receive_ns;
            let value = match sample.applied {
                Some(applied) if !applied.present => Value::Absent,
                _ if stream.image => values::thumbnail(&stream.spec, &sample.payload, sample.seq),
                _ => values::decode(&stream.spec, &stream.names, &sample.payload),
            };
            let series = value.series(3);
            if !series.is_empty() {
                stream.plot.push_back((receive_ns, series));
            }
            let horizon = receive_ns.saturating_sub(ns(config.plot_window));
            while stream.plot.front().is_some_and(|(t, _)| *t < horizon) {
                stream.plot.pop_front();
            }
            stream.value = Some(value);
        }

        if self
            .progress
            .back()
            .is_none_or(|&(t, _)| receive_ns - t >= 100_000_000)
        {
            let newest = self
                .progress
                .back()
                .map_or(sample.capture_ns, |&(_, c)| c.max(sample.capture_ns));
            self.progress.push_back((receive_ns, newest));
            while self.progress.len() > 30 {
                self.progress.pop_front();
            }
        }

        if let Some(applied) = sample.applied {
            self.on_applied(sample, applied, receive_ns);
        }
    }

    fn on_applied(
        &mut self,
        sample: &Sample,
        applied: rlmesh_session::wire::Applied,
        receive_ns: i64,
    ) {
        let Some(part) = self.part_of(sample.channel).cloned() else {
            return;
        };
        let recent = ns(self.config.recent);
        let edge = self.edges.entry(part.channel).or_insert_with(|| EdgeState {
            period_ns: ns(part.control_period()),
            ticks: RateMeter::new(window_for(part.control_hz)),
            last: None,
            last_receive_ns: 0,
            misses: 0,
            recent_misses: VecDeque::new(),
            longest_gap_ns: 0,
            mode: None,
            present: 0,
            total: 0,
        });
        edge.total += 1;
        if applied.present {
            edge.present += 1;
        }
        edge.ticks.record(sample.capture_ns);
        if let Some((seq, at)) = edge.last
            && sample.seq == seq + 1
        {
            // Consecutive ticks only: a seq gap is a loss on the way here,
            // not a late tick.
            let gap = sample.capture_ns - at;
            edge.longest_gap_ns = edge.longest_gap_ns.max(gap);
            if gap > 2 * edge.period_ns {
                edge.misses += 1;
                edge.recent_misses.push_back(receive_ns);
            }
        }
        while edge
            .recent_misses
            .front()
            .is_some_and(|&t| t < receive_ns - recent)
        {
            edge.recent_misses.pop_front();
        }
        edge.last = Some((sample.seq, sample.capture_ns));
        edge.last_receive_ns = receive_ns;
        let changed = edge.mode.is_none_or(|(mode, _)| mode != applied.mode);
        if changed {
            let from = edge.mode.map(|(m, _)| m);
            edge.mode = Some((applied.mode, receive_ns));
            let label = format!("{}/{}", part.owner, part.path);
            let tone = edge_tone(applied.mode);
            self.timeline.edges.entry(label.clone()).or_default().set(
                receive_ns,
                applied.mode.to_string(),
                tone,
            );
            let text = match from {
                Some(from) => format!("edge {label}: {from} -> {}", applied.mode),
                None => format!("edge {label}: {}", applied.mode),
            };
            self.push(receive_ns, Kind::Edge, tone, text);
        }
    }

    fn on_chunk(&mut self, chunk: &ActionChunk, receive_ns: i64) {
        let source_offset = self.offset(&chunk.source);
        let receipt_session_ns = self.session_ns(receive_ns);
        let age = receipt_session_ns - chunk.publish_ns.saturating_add(source_offset);
        let snapshot = self.snapshot.as_ref();
        let fenced = snapshot.is_some_and(|s| {
            s.epoch != chunk.epoch
                || s.lease(chunk.channel)
                    .is_none_or(|l| l.token != chunk.lease_token)
        });
        let dt_ns = if chunk.dt_ns > 0 {
            i64::try_from(chunk.dt_ns).unwrap_or(i64::MAX)
        } else {
            let grid = snapshot
                .and_then(|s| s.binding(&chunk.source))
                .and_then(|b| b.grid(chunk.channel));
            let part = self.part_of(chunk.channel).map(PartSpec::control_period);
            grid.or(part).map_or(1_000_000, ns)
        };
        let recent = ns(self.config.recent);
        let state = self
            .chunks
            .entry(chunk.channel)
            .or_insert_with(|| ChunkState {
                received: RateMeter::new(Duration::from_secs(1)),
                last: None,
                fenced: 0,
            });
        state.received.record(receive_ns);
        if fenced {
            state.fenced += 1;
        } else {
            state.last = Some(LastChunk {
                source: chunk.source.clone(),
                token: chunk.lease_token,
                seq: chunk.seq,
                t0_ns: chunk.t0_ns,
                dt_ns,
                count: chunk.count,
                valid_until_ns: chunk.valid_until_ns,
                receipt_session_ns,
            });
        }
        let window = self.config.counter_window;
        let controller = self
            .controllers
            .entry(chunk.source.clone())
            .or_insert_with(|| ControllerState::new(window));
        controller.chunks.record(receive_ns);
        controller.ages.push_back((receive_ns, age));
        while controller
            .ages
            .front()
            .is_some_and(|&(t, _)| t < receive_ns - recent)
        {
            controller.ages.pop_front();
        }
    }

    fn on_heartbeat(&mut self, heartbeat: Heartbeat, receive_ns: i64) {
        let window = self.config.counter_window;
        for &(channel, seq) in &heartbeat.last_seq_seen {
            self.heard
                .entry((heartbeat.participant.clone(), channel))
                .or_insert_with(|| CounterRate::new(window))
                .record(heartbeat.t1_ns, seq, receive_ns);
        }
        self.heartbeats
            .insert(heartbeat.participant.clone(), (heartbeat, receive_ns));
    }

    fn on_report(&mut self, report: &ParticipantReport, receive_ns: i64) {
        let window = self.config.counter_window;
        for counters in &report.streams {
            let state = self
                .publishers
                .entry(counters.channel)
                .or_insert_with(|| PublisherState {
                    produced: CounterRate::new(window),
                    published: CounterRate::new(window),
                    last: *counters,
                });
            state
                .produced
                .record(report.sent_ns, counters.produced(), receive_ns);
            state
                .published
                .record(report.sent_ns, counters.published, receive_ns);
            state.last = *counters;
        }
        if let Some(run) = report.run {
            let state = self
                .controllers
                .entry(report.participant.clone())
                .or_insert_with(|| ControllerState::new(window));
            state.starts.record(report.sent_ns, run.starts, receive_ns);
            state
                .completions
                .record(report.sent_ns, run.completions, receive_ns);
            state
                .busy
                .record(report.sent_ns, run.busy_skips, receive_ns);
            state.last = Some(run);
        }
    }
}

impl ControllerState {
    fn new(window: Duration) -> Self {
        Self {
            starts: CounterRate::new(window),
            completions: CounterRate::new(window),
            busy: CounterRate::new(window),
            last: None,
            chunks: RateMeter::new(Duration::from_secs(1)),
            ages: VecDeque::new(),
        }
    }
}

/// The tone of an edge mode.
pub fn edge_tone(mode: EdgeMode) -> Tone {
    match mode {
        EdgeMode::Idle => Tone::Off,
        EdgeMode::Driving => Tone::Good,
        EdgeMode::Hold => Tone::Warn,
        EdgeMode::Safe | EdgeMode::Estop => Tone::Bad,
    }
}
