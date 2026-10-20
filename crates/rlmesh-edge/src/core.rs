//! The edge for one actuation channel, sans-IO: chunk intake, the action
//! queue, the fallback ladder and the applied-action log.
//!
//! [`EdgeCore::accept`] validates each chunk on arrival against the edge's
//! last-known authority view and its own clock, and splices it; the
//! embodiment's control loop calls [`EdgeCore::tick`] once per control
//! period. Neither waits on the network: the edge keeps acting on its
//! last-known lease until that lease expires by its own clock, then falls
//! back.
//!
//! # Clock safety
//!
//! Chunk times are session time (the host's clock). With the edge's clock
//! mapping `session = local + θ` within `± ε`:
//!
//! - slot positions (`t0`, `replace_from`) convert as `t − θ`;
//! - **deadlines** (`valid_until`, the lease expiry) convert conservatively,
//!   `t − θ − ε`, so an offset error can only shorten them;
//! - `valid_until` is also clamped to receipt + `max_action_age`, a local age
//!   cap independent of every clock;
//! - with no mapping yet, chunks are refused (`NoClockMapping`) and the lease
//!   cannot be timed (an edge that has applied nothing idles, one that has
//!   falls back), unless [`EdgeConfig::require_clock_mapping`] is off; with
//!   `ε` above [`EdgeConfig::max_clock_error`], chunks are refused
//!   (`ClockUncertain`).

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use rlmesh_wire::{
    ActionChunk, AuthoritySnapshot, ChannelId, ChannelSpec, DType, Direction, Epoch, FenceError,
    LeaseToken, Mode, ParticipantId,
};

use crate::error::EdgeError;
use crate::fallback::{
    EdgeMode, Fallback, FallbackConfig, LeaseView, ModeStats, SafeCause, TickView,
};
use crate::queue::{
    ActionQueue, ChunkMeta, Delivery, QueueStats, QueuedAction, Splice, SpliceOutcome,
};

/// How an edge runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EdgeConfig {
    /// A chunk is never acted on later than this after its receipt, whatever
    /// its `valid_until` says. It also caps how far ahead a chunk's horizon
    /// can reach: size it above the longest horizon controllers send.
    pub max_action_age: Duration,
    /// Ticks kept in the applied-action log (oldest dropped first; 0 keeps
    /// none).
    pub log_capacity: usize,
    /// How actions reach the embodiment ([`Delivery::Once`] for deltas).
    pub delivery: Delivery,
    /// Refuse chunks until this participant has a clock mapping. Turn off
    /// only when every participant shares one clock (one machine, or PTP
    /// with `SystemMonotonicClock`): without a mapping, times are read as
    /// edge time and a delayed snapshot can extend the lease.
    pub require_clock_mapping: bool,
    /// Refuse chunks while the mapping's error bound exceeds this.
    pub max_clock_error: Duration,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            max_action_age: Duration::from_millis(200),
            log_capacity: 4096,
            delivery: Delivery::EveryTick,
            require_clock_mapping: true,
            max_clock_error: Duration::from_millis(50),
        }
    }
}

/// Why the edge refused a chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rejection {
    /// Not a decodable chunk.
    Malformed,
    /// For another channel.
    WrongChannel,
    /// The session is in ESTOP.
    Estop,
    /// From another epoch.
    StaleEpoch,
    /// The channel has no lease.
    NoLease,
    /// Fenced: not the current lease's token (a preempted holder).
    StaleToken,
    /// Its source is not the holder of the lease its token names.
    WrongSource,
    /// The lease expired by the edge's own clock.
    LeaseExpired,
    /// Its seq is not newer than one already accepted under the token.
    StaleSeq,
    /// Already past `valid_until` on arrival.
    Expired,
    /// No actions.
    Empty,
    /// An action does not fit the channel's layout, or is not finite.
    InvalidValue,
    /// No clock mapping yet ([`EdgeConfig::require_clock_mapping`]).
    NoClockMapping,
    /// The clock error bound exceeds [`EdgeConfig::max_clock_error`].
    ClockUncertain,
}

impl Rejection {
    /// Every rejection, in stats order.
    pub const ALL: [Self; 14] = [
        Self::Malformed,
        Self::WrongChannel,
        Self::Estop,
        Self::StaleEpoch,
        Self::NoLease,
        Self::StaleToken,
        Self::WrongSource,
        Self::LeaseExpired,
        Self::StaleSeq,
        Self::Expired,
        Self::Empty,
        Self::InvalidValue,
        Self::NoClockMapping,
        Self::ClockUncertain,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

impl From<FenceError> for Rejection {
    fn from(err: FenceError) -> Self {
        match err {
            FenceError::StaleEpoch { .. } => Self::StaleEpoch,
            FenceError::NoLease { .. } => Self::NoLease,
            FenceError::StaleToken { .. } => Self::StaleToken,
            FenceError::Expired { .. } => Self::LeaseExpired,
        }
    }
}

/// Counts for one edge.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeStats {
    /// Chunks accepted.
    pub accepted: u64,
    /// Chunks refused, by reason (non-zero only).
    pub rejected: Vec<(Rejection, u64)>,
    /// What splicing and draining did to individual actions.
    pub queue: QueueStats,
    /// Entries, ticks and time per edge mode.
    pub modes: Vec<(EdgeMode, ModeStats)>,
    /// Times the edge went safe on its own account (and told the host).
    pub safe_reports: u64,
}

impl EdgeStats {
    /// Chunks refused for `reason`.
    pub fn rejected(&self, reason: Rejection) -> u64 {
        self.rejected
            .iter()
            .find(|(r, _)| *r == reason)
            .map_or(0, |(_, n)| *n)
    }

    /// Chunks refused for any reason.
    pub fn rejected_total(&self) -> u64 {
        self.rejected.iter().map(|(_, n)| n).sum()
    }

    /// Ticks, entries and time in `mode`.
    pub fn mode(&self, mode: EdgeMode) -> ModeStats {
        self.modes
            .iter()
            .find(|(m, _)| *m == mode)
            .map_or_else(ModeStats::default, |(_, s)| *s)
    }
}

/// One control tick, as the edge logs it: what was applied and on whose
/// authority (the action carries its chunk's source, token, seq and
/// `based_on`).
#[derive(Clone, Debug)]
pub struct AppliedAction {
    /// Edge clock at the tick.
    pub at_ns: i64,
    /// The mode.
    pub mode: EdgeMode,
    /// The action applied (`Driving`, or a permitted `Hold` repeat).
    pub action: Option<QueuedAction>,
    /// Whether the safe behaviour ran.
    pub ran_safe: bool,
}

/// What the embodiment does this tick.
#[derive(Clone, Debug)]
pub struct EdgeTick {
    /// The mode.
    pub mode: EdgeMode,
    /// The action to apply, if any.
    pub action: Option<QueuedAction>,
    /// The declared safe behaviour to run, if it must run.
    pub safe_behavior: Option<Arc<str>>,
    /// Set on the tick the edge went safe on its own account, with the
    /// epoch it happened in: the host must be told.
    pub report: Option<(Epoch, SafeCause)>,
}

fn duration_i64(d: Duration) -> i64 {
    i64::try_from(d.as_nanos()).unwrap_or(i64::MAX)
}

/// The edge's clock relation to the session, from the authority view.
#[derive(Clone, Copy, Debug)]
struct ClockView {
    /// session = local + offset.
    offset_ns: i64,
    error_ns: i64,
    mapped: bool,
}

/// The edge for one actuation channel.
#[derive(Debug)]
pub struct EdgeCore {
    channel: ChannelId,
    me: ParticipantId,
    spec: ChannelSpec,
    config: EdgeConfig,
    /// The declared period, for chunks that carry `dt_ns = 0`.
    period_ns: i64,
    safe_behavior: Arc<str>,
    queue: ActionQueue,
    fallback: Fallback,
    /// (epoch, token, seq) of the newest accepted chunk.
    last: Option<(Epoch, LeaseToken, u64)>,
    log: VecDeque<AppliedAction>,
    accepted: u64,
    rejected: [u64; Rejection::ALL.len()],
    safe_reports: u64,
}

impl EdgeCore {
    /// The edge for `spec`, an actuation channel owned by `me`.
    pub fn new(
        spec: ChannelSpec,
        me: ParticipantId,
        config: EdgeConfig,
    ) -> Result<Self, EdgeError> {
        if spec.direction != Direction::Actuate {
            return Err(EdgeError::NotActuation(spec.id));
        }
        if spec.owner.as_ref().is_some_and(|owner| owner != &me) {
            return Err(EdgeError::NotOwner {
                channel: spec.id,
                me,
            });
        }
        let actuation = spec.actuation();
        let period_ns = spec.period().map_or(1, duration_i64).max(1);
        Ok(Self {
            channel: spec.id,
            me,
            period_ns,
            safe_behavior: Arc::from(actuation.safe_behavior.as_str()),
            queue: ActionQueue::new(config.delivery),
            fallback: Fallback::new(FallbackConfig {
                hold_max_ns: duration_i64(actuation.hold_max),
                hold_last: actuation.hold_last,
                delivery: config.delivery,
            }),
            spec,
            config,
            last: None,
            log: VecDeque::with_capacity(config.log_capacity.min(1 << 16)),
            accepted: 0,
            rejected: [0; Rejection::ALL.len()],
            safe_reports: 0,
        })
    }

    /// The channel.
    pub fn channel(&self) -> ChannelId {
        self.channel
    }

    /// The channel's declaration.
    pub fn spec(&self) -> &ChannelSpec {
        &self.spec
    }

    /// Queued actions.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Counts a chunk refused before [`Self::accept`] (one that did not
    /// decode).
    pub fn count_rejection(&mut self, reason: Rejection) {
        self.rejected[reason.index()] += 1;
    }

    fn clock(&self, view: &AuthoritySnapshot) -> ClockView {
        match view.clock(&self.me) {
            Some(mapping) => ClockView {
                offset_ns: mapping.offset_ns,
                error_ns: i64::try_from(mapping.error_ns).unwrap_or(i64::MAX),
                mapped: true,
            },
            None => ClockView {
                offset_ns: 0,
                error_ns: 0,
                mapped: false,
            },
        }
    }

    /// The lease on this channel in the edge clock: the host's expiry
    /// converted conservatively. Unmapped, it is untimeable unless the
    /// mapping is not required, in which case the remaining ttl at snapshot
    /// time counts from when the edge observed the snapshot.
    fn lease_view(&self, view: &AuthoritySnapshot, observed_local_ns: i64) -> Option<LeaseView> {
        let lease = view.lease(self.channel)?;
        let clock = self.clock(view);
        let expires_ns = if clock.mapped {
            Some(
                lease
                    .expires_ns
                    .saturating_sub(clock.offset_ns)
                    .saturating_sub(clock.error_ns),
            )
        } else if self.config.require_clock_mapping {
            None
        } else {
            Some(observed_local_ns.saturating_add(lease.expires_ns.saturating_sub(view.host_ns)))
        };
        Some(LeaseView {
            token: lease.token,
            expires_ns,
        })
    }

    /// Validates `chunk`, received at `receive_ns` (edge clock), against the
    /// view (observed at `observed_local_ns`) and splices it.
    pub fn accept(
        &mut self,
        chunk: ActionChunk,
        receive_ns: i64,
        view: Option<&AuthoritySnapshot>,
        observed_local_ns: i64,
    ) -> Result<SpliceOutcome, Rejection> {
        let result = self.validate_and_splice(chunk, receive_ns, view, observed_local_ns);
        match result {
            Ok(_) => self.accepted += 1,
            Err(reason) => self.rejected[reason.index()] += 1,
        }
        result
    }

    fn validate_and_splice(
        &mut self,
        chunk: ActionChunk,
        receive_ns: i64,
        view: Option<&AuthoritySnapshot>,
        observed_local_ns: i64,
    ) -> Result<SpliceOutcome, Rejection> {
        if chunk.channel != self.channel {
            return Err(Rejection::WrongChannel);
        }
        let view = view.ok_or(Rejection::NoLease)?;
        if view.mode == Mode::Estop {
            return Err(Rejection::Estop);
        }
        let lease = view.check_fence(chunk.channel, chunk.lease_token, chunk.epoch)?;
        if lease.holder != chunk.source {
            return Err(Rejection::WrongSource);
        }
        let clock = self.clock(view);
        if !clock.mapped && self.config.require_clock_mapping {
            return Err(Rejection::NoClockMapping);
        }
        if clock.error_ns > duration_i64(self.config.max_clock_error) {
            return Err(Rejection::ClockUncertain);
        }
        let expires = self
            .lease_view(view, observed_local_ns)
            .and_then(|l| l.expires_ns)
            .ok_or(Rejection::NoLease)?;
        if receive_ns > expires {
            return Err(Rejection::LeaseExpired);
        }
        if chunk.actions.is_empty() {
            return Err(Rejection::Empty);
        }
        if !chunk.actions.iter().all(|a| self.fits(a)) {
            return Err(Rejection::InvalidValue);
        }
        let valid_until_ns = chunk
            .valid_until_ns
            .saturating_sub(clock.offset_ns)
            .saturating_sub(clock.error_ns)
            .min(receive_ns.saturating_add(duration_i64(self.config.max_action_age)));
        if valid_until_ns <= receive_ns {
            return Err(Rejection::Expired);
        }
        let dt_ns = match i64::try_from(chunk.dt_ns) {
            Ok(0) => self.period_ns,
            Ok(dt) => dt,
            Err(_) => return Err(Rejection::Malformed),
        };
        if let Some((epoch, token, seq)) = self.last
            && epoch == chunk.epoch
            && token == chunk.lease_token
            && chunk.seq <= seq
        {
            return Err(Rejection::StaleSeq);
        }
        self.last = Some((chunk.epoch, chunk.lease_token, chunk.seq));
        let meta = Arc::new(ChunkMeta {
            source: chunk.source,
            epoch: chunk.epoch,
            lease_token: chunk.lease_token,
            seq: chunk.seq,
            based_on: chunk.based_on,
            receive_ns,
            valid_until_ns,
        });
        Ok(self.queue.splice(
            Splice {
                chunk: meta,
                t0_ns: chunk.t0_ns.saturating_sub(clock.offset_ns),
                dt_ns,
                replace_from_ns: chunk.replace_from_ns.saturating_sub(clock.offset_ns),
                actions: chunk.actions,
            },
            receive_ns,
        ))
    }

    /// One action against the layout: every leaf's length, and every float
    /// finite (commands are small, so this is always on).
    fn fits(&self, action: &[Bytes]) -> bool {
        self.spec
            .check_leaf_lens(action.iter().map(Bytes::len))
            .is_ok()
            && action
                .iter()
                .zip(&self.spec.layout)
                .all(|(leaf, layout)| finite(leaf, layout.dtype))
    }

    /// One control tick at edge time `now_ns`: what to apply, hold, or the
    /// safe behaviour to run.
    pub fn tick(
        &mut self,
        now_ns: i64,
        view: Option<&AuthoritySnapshot>,
        observed_local_ns: i64,
    ) -> EdgeTick {
        let tick_view = view.map(|v| TickView {
            epoch: v.epoch,
            mode: v.mode,
            lease: self.lease_view(v, observed_local_ns),
        });
        let decision = self.fallback.tick(now_ns, tick_view, &mut self.queue);
        let report = decision
            .went_safe
            .filter(|cause| cause.needs_report())
            .zip(tick_view)
            .map(|(cause, v)| (v.epoch, cause));
        if report.is_some() {
            self.safe_reports += 1;
        }
        if self.config.log_capacity > 0 {
            if self.log.len() >= self.config.log_capacity {
                self.log.pop_front();
            }
            self.log.push_back(AppliedAction {
                at_ns: now_ns,
                mode: decision.mode,
                action: decision.action.clone(),
                ran_safe: decision.run_safe,
            });
        }
        EdgeTick {
            mode: decision.mode,
            safe_behavior: decision.run_safe.then(|| self.safe_behavior.clone()),
            action: decision.action,
            report,
        }
    }

    /// The counters so far.
    pub fn stats(&self) -> EdgeStats {
        EdgeStats {
            accepted: self.accepted,
            rejected: Rejection::ALL
                .iter()
                .map(|r| (*r, self.rejected[r.index()]))
                .filter(|(_, n)| *n > 0)
                .collect(),
            queue: self.queue.stats(),
            modes: self.fallback.stats(),
            safe_reports: self.safe_reports,
        }
    }

    /// Takes the applied-action log: one entry per tick, oldest first.
    pub fn drain_log(&mut self) -> Vec<AppliedAction> {
        self.log.drain(..).collect()
    }
}

/// Whether every element of a little-endian `dtype` leaf is finite.
fn finite(leaf: &[u8], dtype: DType) -> bool {
    match dtype {
        DType::F32 => leaf
            .as_chunks::<4>()
            .0
            .iter()
            .all(|b| f32::from_le_bytes(*b).is_finite()),
        DType::F64 => leaf
            .as_chunks::<8>()
            .0
            .iter()
            .all(|b| f64::from_le_bytes(*b).is_finite()),
        // Exponent all ones is inf or NaN.
        DType::F16 => leaf
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes(*b) & 0x7C00 != 0x7C00),
        DType::Bf16 => leaf
            .as_chunks::<2>()
            .0
            .iter()
            .all(|b| u16::from_le_bytes(*b) & 0x7F80 != 0x7F80),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use rlmesh_session::{Authority, AuthorityConfig};
    use rlmesh_wire::version::PROTOCOL_MINOR;
    use rlmesh_wire::{
        Actuation, ChannelPriority, ChannelQos, ClockMapping, Command, Lease, MergePolicy,
        ObservationRef, PriorityClass, Request, Role, SessionId, TensorLayout,
    };

    use super::*;

    const CMD: ChannelId = ChannelId(2);
    const GRACE: i64 = 100_000_000;
    const MS: i64 = 1_000_000;

    fn pid(id: &str) -> ParticipantId {
        ParticipantId::new(id).expect("id")
    }

    fn cmd_spec(hold_last: bool) -> ChannelSpec {
        let mut spec = ChannelSpec::new(
            CMD,
            "cmd",
            Direction::Actuate,
            "action/joint_pos",
            vec![TensorLayout::f32(&[6])],
            1000.0,
            ChannelQos::realtime(ChannelPriority::Control),
        );
        spec.actuation = Some(Actuation {
            merge: MergePolicy::Replace,
            hold_max: Duration::from_millis(10),
            safe_behavior: "brake".into(),
            hold_last,
        });
        spec
    }

    struct Fixture {
        auth: Authority,
        edge: EdgeCore,
        snapshot: Option<AuthoritySnapshot>,
        observed: i64,
        /// The edge's mapping, when set: offset and error bound.
        mapping: Option<(i64, u64)>,
    }

    impl Fixture {
        fn new() -> Self {
            Self::with(EdgeConfig::default(), true)
        }

        fn with(config: EdgeConfig, hold_last: bool) -> Self {
            let mut auth = Authority::new(
                SessionId::new(),
                AuthorityConfig {
                    lease_grace_ns: GRACE as u64,
                    ..AuthorityConfig::default()
                },
            );
            for (id, roles, priority) in [
                ("robot", vec![Role::Embodiment], PriorityClass::Scripted),
                ("policy", vec![Role::Controller], PriorityClass::Model),
                ("operator", vec![Role::Controller], PriorityClass::Operator),
            ] {
                Self::run(
                    &mut auth,
                    id,
                    Command::Attach {
                        name: id.into(),
                        roles,
                        priority,
                        protocol_minor: PROTOCOL_MINOR,
                        capabilities: Vec::new(),
                    },
                );
            }
            Self::run(
                &mut auth,
                "robot",
                Command::DeclareChannels(vec![cmd_spec(hold_last)]),
            );
            Self::run(&mut auth, "operator", Command::Arm);
            let edge = EdgeCore::new(cmd_spec(hold_last), pid("robot"), config).expect("edge");
            Self {
                auth,
                edge,
                snapshot: None,
                observed: 0,
                mapping: Some((0, 0)),
            }
        }

        fn run(auth: &mut Authority, from: &str, command: Command) {
            auth.apply(
                Request {
                    from: pid(from),
                    epoch: None,
                    command,
                },
                0,
            )
            .expect("apply");
        }

        fn grant(&mut self, from: &str, now: i64) -> Lease {
            self.auth
                .apply(
                    Request {
                        from: pid(from),
                        epoch: None,
                        command: Command::GrantLease {
                            channel: CMD,
                            ttl_ns: None,
                        },
                    },
                    now,
                )
                .expect("grant")
                .lease
                .expect("lease")
        }

        /// The edge observes the host's snapshot taken at `now`.
        fn publish(&mut self, now: i64) {
            let clocks = self
                .mapping
                .map(|(offset_ns, error_ns)| ClockMapping {
                    participant: pid("robot"),
                    offset_ns,
                    error_ns,
                    rtt_ns: 0,
                    sample_ns: now,
                    drift_bound_ppm: 100.0,
                })
                .into_iter()
                .collect();
            self.observed = now;
            self.snapshot = Some(self.auth.snapshot(clocks, now));
        }

        /// `n` actions at 1 ms from `t0` (session time), replacing from `t0`.
        fn chunk(&self, source: &str, lease: &Lease, seq: u64, t0: i64, n: usize) -> ActionChunk {
            ActionChunk {
                channel: CMD,
                source: pid(source),
                lease_token: lease.token,
                epoch: self.auth.epoch(),
                seq,
                publish_ns: t0,
                tick: 0,
                based_on: vec![ObservationRef {
                    channel: ChannelId(1),
                    seq,
                }],
                t0_ns: t0,
                dt_ns: MS as u64,
                replace_from_ns: t0,
                valid_until_ns: t0 + 150 * MS,
                actions: vec![vec![Bytes::from(vec![0_u8; 24])]; n],
            }
        }

        fn accept(&mut self, chunk: ActionChunk, at: i64) -> Result<SpliceOutcome, Rejection> {
            self.edge
                .accept(chunk, at, self.snapshot.as_ref(), self.observed)
        }

        fn tick(&mut self, at: i64) -> EdgeTick {
            self.edge.tick(at, self.snapshot.as_ref(), self.observed)
        }
    }

    #[test]
    fn a_preempted_controllers_queued_and_inflight_chunks_are_fenced() {
        let mut f = Fixture::new();
        let model = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &model, 1, 0, 50), 0)
            .expect("accept");
        assert_eq!(f.tick(MS).mode, EdgeMode::Driving);

        let human = f.grant("operator", 2 * MS);
        f.publish(2 * MS);
        // The model's in-flight chunk is fenced; its queued actions go at the
        // next tick, and the edge waits (braking) until the operator
        // delivers.
        assert_eq!(
            f.accept(f.chunk("policy", &model, 2, 2 * MS, 50), 2 * MS),
            Err(Rejection::StaleToken)
        );
        let tick = f.tick(3 * MS);
        assert_eq!(tick.mode, EdgeMode::Hold);
        assert!(tick.action.is_none() && tick.safe_behavior.is_some());
        // A chunk claiming the operator's token but sent by someone else.
        assert_eq!(
            f.accept(f.chunk("policy", &human, 1, 4 * MS, 50), 4 * MS),
            Err(Rejection::WrongSource)
        );
        f.accept(f.chunk("operator", &human, 1, 4 * MS, 50), 4 * MS)
            .expect("accept");
        let tick = f.tick(4 * MS);
        assert_eq!(tick.mode, EdgeMode::Driving);
        assert_eq!(tick.action.expect("due").chunk.source, pid("operator"));
        let stats = f.edge.stats();
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected(Rejection::StaleToken), 1);
        assert_eq!(stats.rejected(Rejection::WrongSource), 1);
        assert_eq!(stats.queue.fenced, 49, "{stats:?}");

        // The log attributes every tick.
        let log = f.edge.drain_log();
        let sources: Vec<_> = log
            .iter()
            .map(|a| {
                (
                    a.mode,
                    a.action.as_ref().map(|q| q.chunk.source.to_string()),
                    a.ran_safe,
                )
            })
            .collect();
        assert_eq!(
            sources,
            vec![
                (EdgeMode::Driving, Some("policy".into()), false),
                (EdgeMode::Hold, None, true),
                (EdgeMode::Driving, Some("operator".into()), false),
            ]
        );
    }

    #[test]
    fn seq_is_per_token_and_void_or_invalid_chunks_are_rejected() {
        let mut f = Fixture::new();
        let model = f.grant("policy", 0);
        f.publish(0);
        let mut late = f.chunk("policy", &model, 1, 0, 3);
        late.valid_until_ns = 5 * MS;
        assert_eq!(f.accept(late, 6 * MS), Err(Rejection::Expired));
        let mut nan = f.chunk("policy", &model, 1, 0, 3);
        nan.actions[1] = vec![Bytes::from(
            [0.0_f32, f32::NAN, 0.0, 0.0, 0.0, 0.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )];
        assert_eq!(f.accept(nan, 0), Err(Rejection::InvalidValue));
        let mut short = f.chunk("policy", &model, 1, 0, 3);
        short.actions[0] = vec![Bytes::from(vec![0_u8; 20])];
        assert_eq!(f.accept(short, 0), Err(Rejection::InvalidValue));
        for (seq, ok) in [(2, true), (2, false), (1, false)] {
            assert_eq!(
                f.accept(f.chunk("policy", &model, seq, 0, 3), 0).is_ok(),
                ok
            );
        }
        // A new holder's token starts its own seq.
        let human = f.grant("operator", MS);
        f.publish(MS);
        f.accept(f.chunk("operator", &human, 1, MS, 3), MS)
            .expect("accept");
        let stats = f.edge.stats();
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected(Rejection::StaleSeq), 2);
    }

    #[test]
    fn clock_conversion_never_extends_a_deadline() {
        let mut f = Fixture::new();
        // The edge's clock reads 1 s behind session time (offset +1 s),
        // known within 20 ms; edge time 0 is session time 1 s.
        let session_now = 1_000 * MS;
        let lease = f.grant("policy", session_now);
        f.mapping = Some((1_000 * MS, 20 * MS as u64));
        f.publish(session_now);
        let mut chunk = f.chunk("policy", &lease, 1, session_now, 100);
        chunk.valid_until_ns = session_now + 30 * MS;
        f.accept(chunk, 0).expect("accept");
        // valid_until converts to 30 − 20 = 10 ms local: actions from 10 ms on
        // are void, and the queue is dry by 11 ms.
        assert_eq!(f.tick(9 * MS).mode, EdgeMode::Driving);
        let tick = f.tick(11 * MS);
        assert_eq!(
            tick.mode,
            EdgeMode::Hold,
            "void after the conservative deadline"
        );
        assert!(tick.action.is_none(), "an expired action is not repeated");

        // The lease (granted at session 0 with a 100 ms grace) expires at
        // 100 ms session = -900 ms local, less the error bound.
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.mapping = Some((1_000 * MS, 20 * MS as u64));
        f.publish(0);
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 1, 0, 10), -919 * MS),
            Err(Rejection::LeaseExpired)
        );

        // An uncertain or missing mapping refuses chunks.
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.mapping = Some((0, 60 * MS as u64));
        f.publish(0);
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 1, 0, 10), 0),
            Err(Rejection::ClockUncertain)
        );
        f.mapping = None;
        f.publish(0);
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 2, 0, 10), 0),
            Err(Rejection::NoClockMapping)
        );
        assert_eq!(
            f.tick(0).mode,
            EdgeMode::Idle,
            "untimed and nothing applied"
        );
    }

    #[test]
    fn the_lease_lapses_by_the_edge_clock_alone_and_the_edge_reports_safe() {
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        // One snapshot, then the host goes quiet: no renewal ever arrives.
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 140), 0)
            .expect("accept");
        assert_eq!(f.tick(MS).mode, EdgeMode::Driving);
        let tick = f.tick(GRACE + MS);
        assert_eq!(tick.mode, EdgeMode::Safe, "safe at once, no hold");
        assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
        let (epoch, cause) = tick.report.expect("a report for the host");
        assert_eq!((epoch, cause), (f.auth.epoch(), SafeCause::LeaseLapsed));
        // Chunks arriving after the local expiry are refused too.
        assert_eq!(
            f.accept(
                f.chunk("policy", &lease, 2, GRACE + 2 * MS, 10),
                GRACE + 2 * MS
            ),
            Err(Rejection::LeaseExpired)
        );
    }

    #[test]
    fn a_dry_queue_holds_then_goes_safe_and_estop_is_immediate() {
        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 5), 0)
            .expect("accept");
        let ticks: Vec<EdgeTick> = (0..20).map(|t| f.tick(t * MS)).collect();
        let count = |m| ticks.iter().filter(|t| t.mode == m).count();
        assert_eq!(count(EdgeMode::Driving), 5);
        assert_eq!(count(EdgeMode::Hold), 10, "hold_max is 10 ms");
        assert_eq!(count(EdgeMode::Safe), 5);
        assert_eq!(f.edge.stats().safe_reports, 1);
        assert_eq!(
            ticks.iter().filter_map(|t| t.report).collect::<Vec<_>>(),
            vec![(f.auth.epoch(), SafeCause::QueueDry)]
        );

        let mut f = Fixture::new();
        let lease = f.grant("policy", 0);
        f.publish(0);
        f.accept(f.chunk("policy", &lease, 1, 0, 50), 0)
            .expect("accept");
        assert_eq!(f.tick(0).mode, EdgeMode::Driving);
        Fixture::run(
            &mut f.auth,
            "operator",
            Command::Estop {
                reason: "test".into(),
            },
        );
        f.publish(MS);
        let tick = f.tick(MS);
        assert_eq!(tick.mode, EdgeMode::Estop);
        assert_eq!(tick.safe_behavior.as_deref(), Some("brake"));
        assert!(tick.report.is_none(), "the host already knows");
        assert_eq!(
            f.accept(f.chunk("policy", &lease, 2, MS, 50), MS),
            Err(Rejection::Estop)
        );
    }

    #[test]
    fn delta_channels_deliver_each_action_once() {
        let mut f = Fixture::with(
            EdgeConfig {
                delivery: Delivery::Once,
                ..EdgeConfig::default()
            },
            true,
        );
        let lease = f.grant("policy", 0);
        f.publish(0);
        let mut chunk = f.chunk("policy", &lease, 1, 0, 2);
        chunk.dt_ns = 4 * MS as u64;
        f.accept(chunk, 0).expect("accept");
        let delivered: Vec<bool> = (0..10).map(|t| f.tick(t * MS).action.is_some()).collect();
        assert_eq!(
            delivered,
            [
                true, false, false, false, true, false, false, false, false, false
            ]
        );
        assert!(
            f.tick(9 * MS).safe_behavior.is_none(),
            "a delta hold sends nothing"
        );
    }

    #[test]
    fn only_an_actuation_channel_of_its_own_gets_an_edge() {
        let mut sense = cmd_spec(false);
        sense.direction = Direction::Sense;
        sense.actuation = None;
        assert!(matches!(
            EdgeCore::new(sense, pid("robot"), EdgeConfig::default()),
            Err(EdgeError::NotActuation(_))
        ));
        let mut theirs = cmd_spec(false);
        theirs.owner = Some(pid("other"));
        assert!(matches!(
            EdgeCore::new(theirs, pid("robot"), EdgeConfig::default()),
            Err(EdgeError::NotOwner { .. })
        ));
        assert!(!finite(&f32::INFINITY.to_le_bytes(), DType::F32));
        assert!(!finite(&0x7E00_u16.to_le_bytes(), DType::F16), "f16 NaN");
        assert!(finite(&0x3C00_u16.to_le_bytes(), DType::F16), "f16 one");
    }
}
