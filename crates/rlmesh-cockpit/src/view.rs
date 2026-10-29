//! What the UI renders, derived from the [`Model`] at one instant
//! ([`Model::view`]): plain data, so tests check it without a GUI and the UI
//! never touches live state (one snapshot per frame).
//!
//! Every rate names the clock it is counted on ([`clocks`]): the §8.10
//! counters are measured in different places, and a sim slower than real
//! time makes "per second" mean two things.

use std::collections::BTreeSet;
use std::sync::Arc;

use rlmesh_session::server::RECORDER;
use rlmesh_session::wire::{
    BindingRecord, ChannelId, ClockMode, ContractMode, EdgeMode, Health, Mode, PartSpec, Plan,
    PriorityClass, Role,
};
use serde_json::Value as Json;

use crate::model::{Model, Verb, ns};
use crate::timeline::Tone;
use crate::values::{Thumbnail, Value};

/// The clocks rates are counted on.
pub mod clocks {
    /// What the robot declared.
    pub const DECLARED: &str = "declared";
    /// The publisher's own count, per second of its clock.
    pub const PUBLISHER: &str = "publisher clock";
    /// Unique samples arriving here, per second of the cockpit's clock.
    pub const COCKPIT: &str = "cockpit clock";
    /// Samples per second of their capture stamps (session time in virtual
    /// and lockstep modes).
    pub const CAPTURE: &str = "capture time";
    /// The controller runtime's count, per second of its clock.
    pub const CONTROLLER: &str = "controller clock";
    /// The edge's own tick stamps.
    pub const EDGE: &str = "edge ticks";
}

/// Everything the UI shows.
#[derive(Clone, Debug, Default)]
pub struct View {
    /// The session line and the controls' state.
    pub header: Header,
    /// The robot path tree with each part's latest value.
    pub tree: Vec<TreeRow>,
    /// Declared vs measured, per stream.
    pub streams: Vec<StreamRow>,
    /// Per controller.
    pub controllers: Vec<ControllerRow>,
    /// Per edge (commandable part).
    pub edges: Vec<EdgeRow>,
    /// Clock mappings and health per participant.
    pub participants: Vec<ParticipantRow>,
    /// Newest first.
    pub timeline: Vec<TimelineRow>,
    /// Mode, lease and edge lanes.
    pub lanes: Vec<LaneView>,
    /// How far back the lanes reach, in seconds.
    pub lane_window_s: f64,
    /// One per controller binding (§6.8).
    pub bindings: Vec<BindingView>,
}

/// The session line.
#[derive(Clone, Debug, Default)]
pub struct Header {
    /// A snapshot has arrived.
    pub connected: bool,
    /// The host's liveliness token is visible.
    pub host_present: bool,
    /// The session id.
    pub session: String,
    /// How session time advances, and against the cockpit's clock.
    pub clock: String,
    /// The mode.
    pub mode: Option<Mode>,
    /// The epoch.
    pub epoch: u64,
    /// An e-stop is latched.
    pub estop_latched: bool,
    /// The snapshot revision.
    pub revision: u64,
    /// Arm is allowed from here (DISARMED or SAFE).
    pub can_arm: bool,
    /// Disarm is allowed from here.
    pub can_disarm: bool,
    /// What to do about a latched e-stop.
    pub estop_note: Option<String>,
    /// The last verb's answer, and how many seconds ago.
    pub last_outcome: Option<(Verb, Result<(), String>, f64)>,
    /// Inputs the cockpit itself dropped (its UI fell behind).
    pub overrun: u64,
    /// Whether the host records the session.
    pub recording: Recording,
}

/// The host's always-on recorder, as its participant shows it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Recording {
    /// Not in the session (`rlmesh host --no-record`, or another host).
    #[default]
    Off,
    /// Recording.
    On,
    /// Paused: the recordings disk is below its free-space floor.
    Paused,
}

/// One row of the path tree.
#[derive(Clone, Debug, Default)]
pub struct TreeRow {
    /// Nesting depth (0: a robot).
    pub depth: usize,
    /// The last path segment (or the robot id).
    pub label: String,
    /// The part's kind; `None` for a pure path node.
    pub kind: Option<String>,
    /// The stream.
    pub channel: Option<ChannelId>,
    /// Fields, rate and command mode, in brief.
    pub detail: String,
    /// The latest value.
    pub value: Option<Value>,
    /// Now minus capture, in ms.
    pub age_ms: Option<f64>,
    /// Older than its freshness bound.
    pub stale: bool,
    /// Seqs skipped on the way to the cockpit.
    pub lost: u64,
    /// Up to three series of `(seconds ago, value)`, oldest first.
    pub plot: Vec<Vec<[f64; 2]>>,
}

/// Declared vs measured for one stream (§8.10).
#[derive(Clone, Debug, Default)]
pub struct StreamRow {
    /// `robot/path`.
    pub label: String,
    /// Its kind (`_applied` for an applied stream).
    pub kind: String,
    /// The declared source rate; `None` for events.
    pub declared_hz: Option<f64>,
    /// The publication plan: `every`, `latest@D` or `default`.
    pub plan: String,
    /// The `send` ceiling.
    pub ceiling: String,
    /// The admitted contracts: `(controller, contract)`.
    pub admitted: Vec<(String, String)>,
    /// The highest admitted delivery rate.
    pub admitted_hz: Option<f64>,
    /// Produced (publisher clock).
    pub produced_hz: Option<f64>,
    /// Published (publisher clock).
    pub published_hz: Option<f64>,
    /// Unique received here (cockpit clock).
    pub received_hz: Option<f64>,
    /// Received here, per second of capture time.
    pub capture_hz: Option<f64>,
    /// Per subscriber: how fast its newest seen seq advances (its clock).
    pub heard: Vec<(String, Option<f64>)>,
    /// Now minus capture of the newest sample, ms.
    pub age_ms: Option<f64>,
    /// The freshness bound, ms, and where it comes from.
    pub bound: Option<(f64, &'static str)>,
    /// Older than the bound.
    pub stale: bool,
    /// Seqs skipped on the way here.
    pub lost: u64,
    /// `rate_drift`, when the produced (or received) rate is outside
    /// `[0.8, 1.25] ×` the declared rate.
    pub drift: Option<String>,
}

/// One controller (§8.9, §8.10).
#[derive(Clone, Debug, Default)]
pub struct ControllerRow {
    /// Its id.
    pub id: String,
    /// Its lease class.
    pub class: String,
    /// Its binding fingerprint.
    pub binding: Option<String>,
    /// The declared replan rate.
    pub replan_hz: Option<f64>,
    /// No run loop reports (an operator sending by hand, or an older peer).
    pub manual: bool,
    /// Starts per second (controller clock).
    pub starts_hz: Option<f64>,
    /// Completions per second (controller clock).
    pub completions_hz: Option<f64>,
    /// Busy skips, in total.
    pub busy_skips: u64,
    /// Busy skips per second (controller clock).
    pub busy_hz: Option<f64>,
    /// Other skipped slots by reason, nonzero only.
    pub skips: Vec<(&'static str, u64)>,
    /// The inference in flight, ms.
    pub pending_ms: Option<f64>,
    /// Completion latency p50 and p99, ms.
    pub latency_ms: Option<(f64, f64)>,
    /// `rate_drift` on completions.
    pub drifting: bool,
    /// Chunks per second seen here (cockpit clock).
    pub chunks_hz: Option<f64>,
    /// Chunk age at receipt here, newest and recent max, ms.
    pub chunk_age_ms: Option<(f64, f64)>,
    /// Valid coverage remaining on its leased parts (the least), ms.
    pub coverage_ms: Option<f64>,
    /// The parts it leases.
    pub leases: Vec<String>,
}

/// One edge (§9.1, §9.5).
#[derive(Clone, Debug, Default)]
pub struct EdgeRow {
    /// `robot/path`.
    pub label: String,
    /// Its command mode (`position`, `delta`, ...).
    pub command: String,
    /// The declared control rate.
    pub control_hz: f64,
    /// Measured ticks per second (edge ticks).
    pub tick_hz: Option<f64>,
    /// Ticks later than two control periods, in total.
    pub misses: u64,
    /// In the recent window.
    pub recent_misses: usize,
    /// The longest gap between consecutive ticks, ms.
    pub longest_gap_ms: Option<f64>,
    /// The mode now, and for how long, seconds.
    pub mode: Option<(EdgeMode, f64)>,
    /// Ticks that dispatched a value, of all.
    pub present: (u64, u64),
    /// `holder (class) token`.
    pub holder: Option<String>,
    /// Valid coverage remaining, ms (an observer's estimate).
    pub coverage_ms: Option<f64>,
    /// The newest chunk under the current lease: `source token #seq`.
    pub last_chunk: Option<String>,
    /// Chunks under a fenced token or epoch.
    pub fenced: u64,
    /// The robot's reported queue depth.
    pub queue_depth: Option<u32>,
    /// `hold_max`, ms, and the safe behaviour.
    pub hold: (f64, String),
}

/// One participant's clock mapping and health.
#[derive(Clone, Debug, Default)]
pub struct ParticipantRow {
    /// Its id.
    pub id: String,
    /// Its roles.
    pub roles: String,
    /// Its lease class.
    pub class: String,
    /// Session minus participant clock, ms.
    pub offset_ms: Option<f64>,
    /// The mapping's error bound, ms.
    pub error_ms: Option<f64>,
    /// The round trip of the mapping's sample, ms.
    pub rtt_ms: Option<f64>,
    /// Self-reported health.
    pub health: Option<String>,
    /// Since its newest heartbeat arrived here, ms.
    pub heartbeat_age_ms: Option<f64>,
}

/// One timeline entry.
#[derive(Clone, Debug)]
pub struct TimelineRow {
    /// Seconds before now.
    pub ago_s: f64,
    /// Session time, seconds.
    pub session_s: Option<f64>,
    /// The epoch.
    pub epoch: u64,
    /// `mode`, `lease`, ...
    pub kind: &'static str,
    /// Its tone.
    pub tone: Tone,
    /// What happened.
    pub text: String,
}

/// One lane: `(start, end, label, tone)` in seconds relative to now
/// (negative: the past).
#[derive(Clone, Debug)]
pub struct LaneView {
    /// What the lane follows.
    pub label: String,
    /// Its segments, oldest first, clipped to the window.
    pub segments: Vec<(f64, f64, String, Tone)>,
}

/// One controller's binding table (§6.8).
#[derive(Clone, Debug, Default)]
pub struct BindingView {
    /// The first line: `binding {policy} on {robot}: …, fingerprint …`.
    pub title: String,
    /// `replan R Hz`, the revision and whether it holds its leases.
    pub subtitle: String,
    /// Entry rows: `[entry, dir, part, status, plan, delivers]`.
    pub entries: Vec<[String; 6]>,
    /// Input contract rows: `[input, contract, source, delivery, freshness]`.
    pub inputs: Vec<[String; 5]>,
    /// Action rows: `[action, grid, chunk, valid, edge, age cap]`.
    pub actions: Vec<[String; 6]>,
    /// The table as sent, when it is not the binder's JSON.
    pub raw: Option<String>,
}

/// The binder writes `→`; egui's bundled fonts have no glyph for it.
fn ascii_arrows(s: &str) -> String {
    s.replace('→', "->")
}

fn ms(ns: i64) -> f64 {
    ns as f64 * 1e-6
}

fn class_name(p: PriorityClass) -> &'static str {
    match p {
        PriorityClass::Scripted => "scripted",
        PriorityClass::Model => "model",
        PriorityClass::Operator => "operator",
        PriorityClass::Safety => "safety",
    }
}

fn hz_text(hz: f64) -> String {
    if hz <= 0.0 || !hz.is_finite() {
        "event".into()
    } else if hz.fract() == 0.0 {
        format!("{hz:.0} Hz")
    } else {
        format!("{hz:.1} Hz")
    }
}

fn ms_text(v: f64) -> String {
    if v >= 100.0 || (v - v.round()).abs() < 1e-6 {
        format!("{v:.0} ms")
    } else {
        format!("{v:.1} ms")
    }
}

/// The contract in the spec's notation: `latest@15`, `history@15 (10)`,
/// `every`.
pub fn contract_text(mode: ContractMode, hz: f64, history: u32) -> String {
    let d = if hz.fract() == 0.0 {
        format!("{hz:.0}")
    } else {
        format!("{hz:.1}")
    };
    match mode {
        ContractMode::Latest => format!("latest@{d}"),
        ContractMode::History => format!("history@{d} ({history})"),
        ContractMode::Every => "every".into(),
    }
}

impl Model {
    /// Everything the UI shows, at `now_ns` on the cockpit's clock.
    pub fn view(&self, now_ns: i64) -> View {
        let Some(s) = self.snapshot.as_ref() else {
            return View {
                header: Header {
                    host_present: self.host_present,
                    clock: "waiting for the session host".into(),
                    ..Header::default()
                },
                timeline: self.timeline_rows(now_ns),
                lane_window_s: self.config.lane_window.as_secs_f64(),
                ..View::default()
            };
        };
        View {
            header: self.header(now_ns),
            tree: self.tree(now_ns),
            streams: self.stream_rows(now_ns),
            controllers: self.controller_rows(now_ns),
            edges: self.edge_rows(now_ns),
            participants: self.participant_rows(now_ns),
            timeline: self.timeline_rows(now_ns),
            lanes: self.lanes(now_ns),
            lane_window_s: self.lane_window_s(now_ns),
            bindings: s.bindings.iter().map(|b| self.binding_view(b)).collect(),
        }
    }

    fn header(&self, now_ns: i64) -> Header {
        let Some(s) = self.snapshot.as_ref() else {
            return Header::default();
        };
        let clock = match s.clock_mode {
            ClockMode::Wall => "wall clock: session time is the host's monotonic clock".to_string(),
            mode => {
                let name = if mode == ClockMode::Virtual {
                    "virtual"
                } else {
                    "lockstep"
                };
                match self.real_time_factor() {
                    Some(rtf) => format!(
                        "{name} clock: {rtf:.2}× real time (session s per cockpit s), tick {} µs",
                        s.tick_ns / 1000
                    ),
                    None => format!("{name} clock, tick {} µs", s.tick_ns / 1000),
                }
            }
        };
        Header {
            connected: true,
            host_present: self.host_present,
            session: s.session.to_string(),
            clock,
            mode: Some(s.mode),
            epoch: s.epoch.0,
            estop_latched: s.estop_latched,
            revision: s.revision,
            can_arm: matches!(s.mode, Mode::Disarmed | Mode::Safe),
            can_disarm: s.mode != Mode::Disarmed && !(s.mode == Mode::Estop && s.estop_latched),
            estop_note: match (s.mode, s.estop_latched) {
                (Mode::Estop, true) => {
                    Some("e-stop latched: the robot clears it locally, then disarm and arm".into())
                }
                (Mode::Estop, false) => {
                    Some("e-stop cleared at the robot: disarm, then arm".into())
                }
                _ => None,
            },
            last_outcome: self
                .last_outcome
                .clone()
                .map(|(verb, result, at)| (verb, result, ms(now_ns - at) * 1e-3)),
            overrun: self.overrun,
            recording: match s.roster.iter().find(|p| p.id.as_str() == RECORDER) {
                None => Recording::Off,
                Some(p) => match self.heartbeats.get(&p.id) {
                    Some((hb, _)) if hb.health == Health::Degraded => Recording::Paused,
                    _ => Recording::On,
                },
            },
        }
    }

    /// Session seconds per cockpit second, over the recent samples.
    fn real_time_factor(&self) -> Option<f64> {
        let (&(t0, c0), &(t1, c1)) = (self.progress.front()?, self.progress.back()?);
        (t1 - t0 > 500_000_000).then(|| (c1 - c0) as f64 / (t1 - t0) as f64)
    }

    /// The freshness bound of a stream (§8.7): its contracts' effective
    /// bound, else the robot's declared one.
    fn bound(&self, channel: ChannelId, part: &PartSpec) -> Option<(f64, &'static str)> {
        let s = self.snapshot.as_ref()?;
        let contracted = s
            .bindings
            .iter()
            .flat_map(|b| &b.contracts)
            .filter(|c| c.channel == channel)
            .filter_map(|c| c.max_age.map(|age| (age, c.max_age_explicit)))
            .min();
        match contracted {
            Some((age, true)) => Some((ms(ns(age)), "explicit")),
            Some((age, false)) => Some((ms(ns(age)), "derived")),
            None => part
                .max_age
                .filter(|_| channel == part.channel)
                .map(|age| (ms(ns(age)), "robot")),
        }
    }

    fn age_ms(&self, channel: ChannelId, now_ns: i64) -> Option<f64> {
        let (_, capture, _) = self.streams.get(&channel)?.last?;
        Some(ms(now_ns - capture))
    }

    fn tree(&self, now_ns: i64) -> Vec<TreeRow> {
        let Some(s) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        let mut parts: Vec<&PartSpec> = s.parts.iter().collect();
        parts.sort_by(|a, b| (&a.owner, a.path.as_str()).cmp(&(&b.owner, b.path.as_str())));
        let mut rows = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        for part in parts {
            let robot = part.owner.to_string();
            if seen.insert(robot.clone()) {
                rows.push(TreeRow {
                    depth: 0,
                    label: robot.clone(),
                    detail: s
                        .contract
                        .get(&part.owner)
                        .map(|f| format!("declaration {f}"))
                        .unwrap_or_default(),
                    ..TreeRow::default()
                });
            }
            let segments: Vec<&str> = part.path.as_str().split('/').collect();
            for depth in 1..segments.len() {
                let prefix = format!("{robot}/{}", segments[..depth].join("/"));
                if seen.insert(prefix) {
                    rows.push(TreeRow {
                        depth,
                        label: segments[depth - 1].to_string(),
                        ..TreeRow::default()
                    });
                }
            }
            seen.insert(format!("{robot}/{}", part.path));
            let fields: Vec<String> = part
                .fields
                .iter()
                .map(|f| format!("{} {}{:?}", f.name, f.dtype, f.shape))
                .collect();
            let mut detail = format!("{} · {}", fields.join(", "), hz_text(part.hz));
            if let Some(c) = &part.command {
                detail.push_str(&format!(
                    " · command {} @ {}",
                    c.mode,
                    hz_text(part.control_hz)
                ));
            }
            rows.push(self.tree_row(part, part.channel, segments.len(), detail, now_ns));
            if let Some(applied) = part.applied_channel {
                let mode = part.command.as_ref().map_or("", |c| c.mode.as_str());
                rows.push(self.tree_row(
                    part,
                    applied,
                    segments.len() + 1,
                    format!("what the edge dispatched ({mode}), one per control tick"),
                    now_ns,
                ));
            }
        }
        rows
    }

    fn tree_row(
        &self,
        part: &PartSpec,
        channel: ChannelId,
        depth: usize,
        detail: String,
        now_ns: i64,
    ) -> TreeRow {
        let applied = channel != part.channel;
        let stream = self.streams.get(&channel);
        let age_ms = self.age_ms(channel, now_ns);
        let stale = match (age_ms, self.bound(channel, part)) {
            (Some(age), Some((bound, _))) => age > bound,
            _ => false,
        };
        let plot = stream.map_or_else(Vec::new, |st| {
            let n = st.plot.back().map_or(0, |(_, v)| v.len());
            (0..n)
                .map(|i| {
                    st.plot
                        .iter()
                        .filter_map(|(t, v)| v.get(i).map(|v| [ms(*t - now_ns) * 1e-3, *v]))
                        .collect()
                })
                .collect()
        });
        TreeRow {
            depth,
            label: if applied {
                "_applied".into()
            } else {
                part.path
                    .as_str()
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_string()
            },
            kind: Some(if applied {
                "applied".into()
            } else {
                part.kind.clone()
            }),
            channel: Some(channel),
            detail,
            value: stream.and_then(|st| st.value.clone()),
            age_ms,
            stale,
            lost: stream.map_or(0, |st| st.lost),
            plot,
        }
    }

    fn stream_rows(&self, now_ns: i64) -> Vec<StreamRow> {
        let Some(s) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        let mut rows = Vec::new();
        for part in &s.parts {
            let mut channels = vec![(part.channel, false)];
            channels.extend(part.applied_channel.map(|c| (c, true)));
            for (channel, applied) in channels {
                let declared_hz = if applied {
                    Some(part.control_hz)
                } else {
                    (!part.event && part.hz > 0.0).then_some(part.hz)
                };
                let contracts: Vec<(String, String, f64)> = s
                    .bindings
                    .iter()
                    .flat_map(|b| b.contracts.iter().map(move |c| (b, c)))
                    .filter(|(_, c)| c.channel == channel)
                    .map(|(b, c)| {
                        (
                            b.participant.to_string(),
                            contract_text(c.mode, c.delivery_hz, c.history),
                            c.delivery_hz,
                        )
                    })
                    .collect();
                let plan = match s.demand(channel).map(|d| d.plan) {
                    Some(Plan::Every) => "every".to_string(),
                    Some(Plan::Latest { hz }) => format!("latest@{hz:.0}"),
                    None => "default (every within the ceiling)".to_string(),
                };
                let publisher = self.publishers.get(&channel);
                let stream = self.streams.get(&channel);
                let produced_hz = publisher.and_then(|p| p.produced.rate(now_ns));
                let received_hz = stream.and_then(|st| st.received.rate(now_ns));
                let drift = declared_hz.and_then(|declared| {
                    let (measured, what) = match (produced_hz, received_hz) {
                        (Some(p), _) => (p, "produced"),
                        (None, Some(r)) => (r, "received"),
                        _ => return None,
                    };
                    let ratio = measured / declared;
                    (!(0.8..=1.25).contains(&ratio)).then(|| {
                        format!("rate_drift: {what} {measured:.1} of {declared} Hz declared")
                    })
                });
                let bound = self.bound(channel, part);
                let age_ms = self.age_ms(channel, now_ns);
                let heard = self
                    .heard
                    .iter()
                    .filter(|((who, c), _)| *c == channel && who != &self.me)
                    .map(|((who, _), rate)| (who.to_string(), rate.rate(now_ns)))
                    .collect();
                rows.push(StreamRow {
                    label: if applied {
                        format!("{}/{}/_applied", part.owner, part.path)
                    } else {
                        format!("{}/{}", part.owner, part.path)
                    },
                    kind: if applied {
                        "_applied".into()
                    } else {
                        part.kind.clone()
                    },
                    declared_hz,
                    plan,
                    ceiling: if applied {
                        "every".into()
                    } else {
                        part.send.clone()
                    },
                    admitted_hz: contracts.iter().map(|c| c.2).reduce(f64::max),
                    admitted: contracts.into_iter().map(|(w, c, _)| (w, c)).collect(),
                    produced_hz,
                    published_hz: publisher.and_then(|p| p.published.rate(now_ns)),
                    received_hz,
                    capture_hz: stream.and_then(|st| st.captured.rate(st.captured_end(now_ns))),
                    heard,
                    age_ms,
                    stale: matches!((age_ms, bound), (Some(a), Some((b, _))) if a > b),
                    bound,
                    lost: stream.map_or(0, |st| st.lost),
                    drift,
                });
            }
        }
        rows.sort_by(|a, b| a.label.cmp(&b.label));
        rows
    }

    /// Valid coverage remaining on a commandable part (§9.2), as an
    /// observer estimates it: `min(t0 + horizon·dt, valid_until, receipt +
    /// max_action_age, lease expiry) − now`, in session time, with the
    /// cockpit's receipt standing in for the edge's.
    fn coverage_ms(&self, channel: ChannelId, now_ns: i64) -> Option<f64> {
        let s = self.snapshot.as_ref()?;
        let lease = s.lease(channel)?;
        let chunk = self.chunks.get(&channel)?.last.as_ref()?;
        if chunk.token != lease.token {
            return Some(0.0);
        }
        let max_age = s
            .part(channel)
            .and_then(|p| p.command.as_ref())
            .map_or(i64::MAX, |c| ns(c.max_action_age));
        let end = chunk
            .t0_ns
            .saturating_add(chunk.dt_ns.saturating_mul(i64::from(chunk.count)))
            .min(chunk.valid_until_ns)
            .min(chunk.receipt_session_ns.saturating_add(max_age))
            .min(lease.expires_ns);
        Some(ms((end - self.session_ns(now_ns)).max(0)))
    }

    fn controller_rows(&self, now_ns: i64) -> Vec<ControllerRow> {
        let Some(s) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        let mut rows = Vec::new();
        for p in s.roster.iter().filter(|p| p.has_role(Role::Controller)) {
            let binding = s.binding(&p.id);
            let state = self.controllers.get(&p.id);
            let run = state.and_then(|c| c.last);
            let leased: Vec<ChannelId> = s
                .leases
                .iter()
                .filter(|l| l.holder == p.id)
                .map(|l| l.channel)
                .collect();
            let ages = state.map(|c| &c.ages);
            let newest_age = ages.and_then(|a| a.back()).map(|&(_, age)| age);
            let max_age = ages.and_then(|a| a.iter().map(|&(_, age)| age).max());
            rows.push(ControllerRow {
                id: p.id.to_string(),
                class: class_name(p.priority).into(),
                binding: binding.map(|b| b.fingerprint.to_string()),
                replan_hz: binding.map(|b| b.replan_hz).filter(|hz| *hz > 0.0),
                manual: run.is_none_or(|r| r.starts == 0 && r.completions == 0),
                starts_hz: state.and_then(|c| c.starts.rate(now_ns)),
                completions_hz: state.and_then(|c| c.completions.rate(now_ns)),
                busy_skips: run.map_or(0, |r| r.busy_skips),
                busy_hz: state.and_then(|c| c.busy.rate(now_ns)),
                skips: run.map_or_else(Vec::new, |r| {
                    [
                        ("waiting_for", r.waiting_for),
                        ("stale_input", r.stale_input),
                        ("window_gap", r.window_gap),
                        ("decode", r.decode),
                        ("no_lease", r.no_lease),
                    ]
                    .into_iter()
                    .filter(|(_, n)| *n > 0)
                    .collect()
                }),
                pending_ms: run
                    .filter(|r| r.pending_ns > 0)
                    .map(|r| r.pending_ns as f64 * 1e-6),
                latency_ms: run.filter(|r| r.latency_p50_ns > 0).map(|r| {
                    (
                        r.latency_p50_ns as f64 * 1e-6,
                        r.latency_p99_ns as f64 * 1e-6,
                    )
                }),
                drifting: run.is_some_and(|r| r.drifting),
                chunks_hz: state.and_then(|c| c.chunks.rate(now_ns)),
                chunk_age_ms: newest_age.map(|newest| (ms(newest), ms(max_age.unwrap_or(newest)))),
                coverage_ms: leased
                    .iter()
                    .filter_map(|c| self.coverage_ms(*c, now_ns))
                    .reduce(f64::min),
                leases: leased
                    .iter()
                    .filter_map(|c| s.part(*c))
                    .map(|part| format!("{}/{}", part.owner, part.path))
                    .collect(),
            });
        }
        rows
    }

    fn edge_rows(&self, now_ns: i64) -> Vec<EdgeRow> {
        let Some(s) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        s.parts
            .iter()
            .filter_map(|part| part.command.as_ref().map(|c| (part, c)))
            .map(|(part, command)| {
                let edge = self.edges.get(&part.channel);
                let lease = s.lease(part.channel);
                EdgeRow {
                    label: format!("{}/{}", part.owner, part.path),
                    command: command.mode.clone(),
                    control_hz: part.control_hz,
                    tick_hz: edge.and_then(|e| e.ticks.rate(e.ticks_end(now_ns))),
                    misses: edge.map_or(0, |e| e.misses),
                    recent_misses: edge.map_or(0, |e| e.recent_misses.len()),
                    longest_gap_ms: edge
                        .filter(|e| e.longest_gap_ns > 0)
                        .map(|e| ms(e.longest_gap_ns)),
                    mode: edge
                        .and_then(|e| e.mode)
                        .map(|(mode, since)| (mode, ms(now_ns - since) * 1e-3)),
                    present: edge.map_or((0, 0), |e| (e.present, e.total)),
                    holder: lease
                        .map(|l| format!("{} ({}) {}", l.holder, class_name(l.priority), l.token)),
                    coverage_ms: self.coverage_ms(part.channel, now_ns),
                    last_chunk: self
                        .chunks
                        .get(&part.channel)
                        .and_then(|c| c.last.as_ref())
                        .map(|c| format!("{} {} #{}", c.source, c.token, c.seq)),
                    fenced: self.chunks.get(&part.channel).map_or(0, |c| c.fenced),
                    queue_depth: self
                        .heartbeats
                        .get(&part.owner)
                        .map(|(hb, _)| hb.queue_depth),
                    hold: (ms(ns(command.hold_max)), command.safe_behavior.clone()),
                }
            })
            .collect()
    }

    fn participant_rows(&self, now_ns: i64) -> Vec<ParticipantRow> {
        let Some(s) = self.snapshot.as_ref() else {
            return Vec::new();
        };
        s.roster
            .iter()
            .map(|p| {
                let clock = s.clock(&p.id);
                let hb = self.heartbeats.get(&p.id);
                ParticipantRow {
                    id: if p.id == self.me {
                        format!("{} (this cockpit)", p.id)
                    } else {
                        p.id.to_string()
                    },
                    roles: p
                        .roles
                        .iter()
                        .map(|r| format!("{r:?}").to_lowercase())
                        .collect::<Vec<_>>()
                        .join(", "),
                    class: class_name(p.priority).into(),
                    offset_ms: clock.map(|c| ms(c.offset_ns)),
                    error_ms: clock.map(|c| c.error_ns as f64 * 1e-6),
                    rtt_ms: clock.map(|c| c.rtt_ns as f64 * 1e-6),
                    health: hb.map(|(hb, _)| format!("{:?}", hb.health).to_lowercase()),
                    heartbeat_age_ms: hb.map(|(_, at)| ms(now_ns - at)),
                }
            })
            .collect()
    }

    fn timeline_rows(&self, now_ns: i64) -> Vec<TimelineRow> {
        self.timeline
            .entries()
            .rev()
            .map(|e| TimelineRow {
                ago_s: ms(now_ns - e.at_ns) * 1e-3,
                session_s: e.session_ns.map(|t| t as f64 * 1e-9),
                epoch: e.epoch,
                kind: e.kind.label(),
                tone: e.tone,
                text: e.text.clone(),
            })
            .collect()
    }

    /// The lanes' reach: the configured window, or less while the cockpit
    /// has seen less (at least 10 s).
    fn lane_window_s(&self, now_ns: i64) -> f64 {
        let first = self
            .timeline
            .mode
            .segments()
            .next()
            .map_or(now_ns, |s| s.start_ns);
        (ms(now_ns - first) * 1e-3).clamp(10.0, self.config.lane_window.as_secs_f64().max(10.0))
    }

    fn lanes(&self, now_ns: i64) -> Vec<LaneView> {
        let window = self.lane_window_s(now_ns);
        let lane = |label: String, lane: &crate::timeline::Lane| {
            let segments: Vec<_> = lane.segments().collect();
            LaneView {
                label,
                segments: segments
                    .iter()
                    .enumerate()
                    .map(|(i, seg)| {
                        let end = segments.get(i + 1).map_or(now_ns, |next| next.start_ns);
                        let rel = |t: i64| (ms(t - now_ns) * 1e-3).max(-window);
                        (rel(seg.start_ns), rel(end), seg.label.clone(), seg.tone)
                    })
                    .collect(),
            }
        };
        let mut lanes = vec![lane("session mode".into(), &self.timeline.mode)];
        for (label, l) in &self.timeline.leases {
            lanes.push(lane(format!("lease {label}"), l));
        }
        for (label, l) in &self.timeline.edges {
            lanes.push(lane(format!("edge {label}"), l));
        }
        lanes
    }

    fn binding_view(&self, b: &BindingRecord) -> BindingView {
        let Some(s) = self.snapshot.as_ref() else {
            return BindingView::default();
        };
        let holds = s.leases.iter().any(|l| l.holder == b.participant);
        let subtitle = format!(
            "{} · revision {} · replan {} · {}",
            b.participant,
            b.revision,
            if b.replan_hz > 0.0 {
                hz_text(b.replan_hz)
            } else {
                "none (commands by hand)".into()
            },
            if holds {
                "holds its leases"
            } else {
                "no lease held (standby)"
            }
        );
        let Ok(table) = serde_json::from_str::<Json>(&b.table_json) else {
            return BindingView {
                title: format!("binding {} by {}", b.fingerprint, b.participant),
                subtitle,
                raw: Some(b.table_json.clone()),
                ..BindingView::default()
            };
        };
        let text = |v: &Json, key: &str| match v.get(key) {
            Some(Json::String(s)) => ascii_arrows(s),
            Some(Json::Null) | None => "-".into(),
            Some(other) => other.to_string(),
        };
        let rows = |key: &str| -> Vec<Json> {
            table
                .get(key)
                .and_then(Json::as_array)
                .cloned()
                .unwrap_or_default()
        };
        let (inputs, actions) = (rows("inputs"), rows("actions"));
        if table.get("inputs").is_none() && table.get("actions").is_none() {
            // An identity binding (§6.5): parts commanded by path, no plan.
            let parts: Vec<String> = b
                .channels
                .iter()
                .filter_map(|c| s.part(*c))
                .map(|p| p.path.to_string())
                .collect();
            return BindingView {
                title: format!(
                    "identity binding by {}: commands {}, fingerprint {}",
                    b.participant,
                    parts.join(", "),
                    b.fingerprint
                ),
                subtitle,
                entries: parts
                    .iter()
                    .map(|p| {
                        [
                            p.clone(),
                            "out".into(),
                            p.clone(),
                            "identity".into(),
                            "none".into(),
                            "-".into(),
                        ]
                    })
                    .collect(),
                ..BindingView::default()
            };
        }
        let plan = |v: &Json| {
            let steps: Vec<String> = v
                .get("plan")
                .and_then(Json::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|s| s.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            if steps.is_empty() {
                "none".into()
            } else {
                ascii_arrows(&steps.join(" -> "))
            }
        };
        let mut entries = Vec::new();
        for (dir, list) in [("in", &inputs), ("out", &actions)] {
            for row in list {
                entries.push([
                    text(row, "entry"),
                    dir.into(),
                    text(row, "part"),
                    text(row, "status"),
                    plan(row),
                    text(row, "delivers"),
                ]);
            }
        }
        let input_rows = inputs
            .iter()
            .map(|row| {
                let channel = row.get("channel").and_then(Json::as_u64);
                let contract =
                    channel.and_then(|c| b.contracts.iter().find(|k| u64::from(k.channel.0) == c));
                let source = channel
                    .and_then(|c| s.stream(ChannelId(u32::try_from(c).unwrap_or(u32::MAX))))
                    .and_then(|st| self.part_of(st.channel).map(|p| (p.clone(), st.applied)))
                    .map_or("-".into(), |(p, applied)| {
                        hz_text(if applied { p.control_hz } else { p.hz })
                    });
                [
                    text(row, "entry"),
                    contract.map_or("-".into(), |c| {
                        contract_text(c.mode, c.delivery_hz, c.history)
                    }),
                    source,
                    contract.map_or("-".into(), |c| hz_text(c.delivery_hz)),
                    contract
                        .and_then(|c| c.max_age.map(|age| (age, c.max_age_explicit)))
                        .map_or("-".into(), |(age, explicit)| {
                            format!(
                                "{} ({})",
                                ms_text(ms(ns(age))),
                                if explicit { "explicit" } else { "derived" }
                            )
                        }),
                ]
            })
            .collect();
        let action_rows = actions
            .iter()
            .map(|row| {
                let part = row
                    .get("channel")
                    .and_then(Json::as_u64)
                    .and_then(|c| s.part(ChannelId(u32::try_from(c).unwrap_or(u32::MAX))));
                let dt_ns = row.get("dt_ns").and_then(Json::as_u64).unwrap_or(0);
                let horizon = row.get("horizon").and_then(Json::as_u64).unwrap_or(0);
                let valid_ns = row.get("valid_ns").and_then(Json::as_u64).unwrap_or(0);
                let dt_ms = dt_ns as f64 * 1e-6;
                [
                    text(row, "entry"),
                    if dt_ns > 0 {
                        hz_text(1e9 / dt_ns as f64)
                    } else {
                        "-".into()
                    },
                    format!(
                        "{horizon} × {} = {}",
                        ms_text(dt_ms),
                        ms_text(dt_ms * horizon as f64)
                    ),
                    ms_text(valid_ns as f64 * 1e-6),
                    part.map_or("-".into(), |p| hz_text(p.control_hz)),
                    part.and_then(|p| p.command.as_ref())
                        .map_or("-".into(), |c| ms_text(ms(ns(c.max_action_age)))),
                ]
            })
            .collect();
        BindingView {
            title: format!(
                "binding {} on {}: {} inputs, {} actions, fingerprint {}",
                text(&table, "policy"),
                text(&table, "robot"),
                inputs.len(),
                actions.len(),
                b.fingerprint
            ),
            subtitle,
            entries,
            inputs: input_rows,
            actions: action_rows,
            raw: None,
        }
    }
}

impl crate::model::StreamState {
    /// Now on the publisher's capture clock: the newest stamp plus the
    /// cockpit time since it arrived, so silence counts on any clock.
    pub(crate) fn captured_end(&self, now_ns: i64) -> i64 {
        let received = self.last.map_or(now_ns, |(_, _, at)| at);
        self.captured_newest
            .saturating_add((now_ns - received).max(0))
    }
}

impl crate::model::EdgeState {
    /// Now on the edge's tick clock, as [`StreamState::captured_end`].
    ///
    /// [`StreamState::captured_end`]: crate::model::StreamState::captured_end
    pub(crate) fn ticks_end(&self, now_ns: i64) -> i64 {
        self.last
            .map_or(0, |(_, at)| at)
            .saturating_add((now_ns - self.last_receive_ns).max(0))
    }
}

/// A thumbnail and the seq it shows, for texture caching in the UI.
pub fn thumbnail_of(value: &Value) -> Option<&Arc<Thumbnail>> {
    match value {
        Value::Image(t) => Some(t),
        _ => None,
    }
}
