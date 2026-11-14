//! What an edge runtime needs from its participant: the authority view, the
//! edge report, the local e-stop latch and the ordered stop queue.
//!
//! A robot process tells the host about its own stops through **one ordered
//! queue**: `EnterSafe`, a local `Estop` and its `Clear` are
//! sent in the order they happened, each retried until the host answers, so
//! a host that comes back after a partition sees `Estop` before `Clear` and
//! ends in ESTOP (unlatched), which still needs disarm and arm.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::wire::{
    AuthoritySnapshot, ChunkTiming, ClockRefusal, Command, CtlStatus, Epoch, Mode, ParticipantId,
    Reason, ReasonCode,
};
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::edge::{EdgeCore, Rejection};
use crate::error::SessionError;
#[cfg(doc)]
use crate::participant::ParticipantConfig;
use crate::participant::{Participant, Shared, lock};

/// What an edge reports in its participant's heartbeats.
#[derive(Debug, Default)]
pub struct EdgeReport {
    last_accepted: AtomicU64,
    last_applied: AtomicU64,
    queue_depth: AtomicU32,
    /// Chunks refused, by reason in [`Rejection::ALL`] order, over every
    /// edge of the robot.
    rejected: [AtomicU64; Rejection::ALL.len()],
    /// One per edge whose newest clock judgement refused a chunk.
    clock_refused: Mutex<Vec<ClockRefusal>>,
    /// Chunks accepted, over every edge of the robot.
    accepted: AtomicU64,
    /// How each commanded part's chunks arrived, over every edge.
    timing: Mutex<Vec<ChunkTiming>>,
}

impl EdgeReport {
    /// A chunk accepted, with `seq`: the newest.
    pub fn set_accepted(&self, seq: u64) {
        self.last_accepted.store(seq, Ordering::Relaxed);
        self.accepted.fetch_add(1, Ordering::Relaxed);
    }

    /// Chunks accepted so far.
    pub fn accepted(&self) -> u64 {
        self.accepted.load(Ordering::Relaxed)
    }

    /// Records how `edge`'s members' chunks arrived.
    pub fn set_timing(&self, edge: &EdgeCore) {
        let mut timing = lock(&self.timing);
        for member in edge.timing() {
            match timing.iter_mut().find(|t| t.channel == member.channel) {
                Some(held) => *held = *member,
                None => timing.push(*member),
            }
        }
    }

    /// How each commanded part's chunks arrived so far.
    pub fn timing(&self) -> Vec<ChunkTiming> {
        lock(&self.timing).clone()
    }

    /// The newest applied chunk's seq.
    pub fn set_applied(&self, seq: u64) {
        self.last_applied.store(seq, Ordering::Relaxed);
    }

    /// Queued actions.
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth
            .store(u32::try_from(depth).unwrap_or(u32::MAX), Ordering::Relaxed);
    }

    /// One chunk refused for `reason`.
    pub fn count_rejection(&self, reason: Rejection) {
        self.rejected[reason.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Chunks refused so far, by reason (nonzero only).
    pub fn rejected(&self) -> Vec<(Rejection, u64)> {
        Rejection::ALL
            .iter()
            .map(|r| (*r, self.rejected[r.index()].load(Ordering::Relaxed)))
            .filter(|(_, n)| *n > 0)
            .collect()
    }

    /// Records `edge`'s newest clock judgement.
    pub fn set_clock_refused(&self, edge: &EdgeCore) {
        let mut refused = lock(&self.clock_refused);
        refused.retain(|r| !edge.parts().any(|p| p.channel == r.channel));
        refused.extend(edge.clock_refused().cloned());
    }

    /// One per edge whose newest clock judgement refused a chunk.
    pub fn clock_refused(&self) -> Vec<ClockRefusal> {
        lock(&self.clock_refused).clone()
    }

    /// `(last accepted, last applied, queue depth)`.
    pub fn get(&self) -> (u64, u64, u32) {
        (
            self.last_accepted.load(Ordering::Relaxed),
            self.last_applied.load(Ordering::Relaxed),
            self.queue_depth.load(Ordering::Relaxed),
        )
    }
}

/// A participant's live view of the authority state, for code that runs off
/// the async path (an edge's control loop). Cheap to clone.
#[derive(Clone, Debug)]
pub struct AuthorityView {
    pub(super) rx: watch::Receiver<Option<AuthoritySnapshot>>,
    pub(super) observed_local_ns: Arc<AtomicI64>,
    pub(super) epoch: Arc<AtomicU64>,
}

impl AuthorityView {
    /// Runs `f` on the newest snapshot without cloning it.
    pub fn with<R>(&self, f: impl FnOnce(Option<&AuthoritySnapshot>) -> R) -> R {
        f(self.rx.borrow().as_ref())
    }

    /// A copy of the newest snapshot.
    pub fn current(&self) -> Option<AuthoritySnapshot> {
        self.rx.borrow().clone()
    }

    /// This participant's clock when it took the newest snapshot.
    pub fn observed_local_ns(&self) -> i64 {
        self.observed_local_ns.load(Ordering::Relaxed)
    }

    /// The newest snapshot's epoch (0 before the first).
    pub fn epoch(&self) -> Epoch {
        Epoch(self.epoch.load(Ordering::Relaxed))
    }

    /// A receiver that wakes on every new snapshot.
    pub fn subscribe(&self) -> watch::Receiver<Option<AuthoritySnapshot>> {
        self.rx.clone()
    }

    /// A view fed by hand (tests and sans-IO users).
    pub fn manual() -> (ManualView, Self) {
        let (tx, rx) = watch::channel(None);
        let observed_local_ns = Arc::new(AtomicI64::new(0));
        let epoch = Arc::new(AtomicU64::new(0));
        (
            ManualView {
                tx,
                observed_local_ns: observed_local_ns.clone(),
                epoch: epoch.clone(),
            },
            Self {
                rx,
                observed_local_ns,
                epoch,
            },
        )
    }
}

/// The feeding end of [`AuthorityView::manual`].
#[derive(Debug)]
pub struct ManualView {
    tx: watch::Sender<Option<AuthoritySnapshot>>,
    observed_local_ns: Arc<AtomicI64>,
    epoch: Arc<AtomicU64>,
}

impl ManualView {
    /// Publishes `snapshot`, observed at `local_ns`.
    pub fn set(&self, snapshot: AuthoritySnapshot, local_ns: i64) {
        self.observed_local_ns.store(local_ns, Ordering::Relaxed);
        self.epoch.store(snapshot.epoch.0, Ordering::Relaxed);
        self.tx.send_replace(Some(snapshot));
    }
}

/// A robot process's local e-stop latch: an atomic flag, and how many
/// times it was latched, so an edge that missed a latch cleared before its
/// next look still stops (and fences) for it.
#[derive(Debug, Default)]
pub(crate) struct LocalLatch {
    latched: AtomicBool,
    latches: AtomicU64,
    /// Why it latched last: `ESTOP_LOCAL` (its own stop) or
    /// `ESTOP_SIGNAL` (a member's signal), and by whom.
    reason: Mutex<Reason>,
}

impl LocalLatch {
    /// Latches it for `reason`: which latch this is. The reason, then the
    /// count, then the flag, so whoever sees the flag also sees both.
    pub(super) fn latch(&self, reason: Reason) -> u64 {
        let mut last = lock(&self.reason);
        *last = reason;
        let seq = self.latches.fetch_add(1, Ordering::SeqCst) + 1;
        self.latched.store(true, Ordering::SeqCst);
        seq
    }

    /// Clears it: how many latches came before, and why the last one
    /// latched.
    fn clear(&self) -> (u64, Reason) {
        let last = lock(&self.reason);
        self.latched.store(false, Ordering::SeqCst);
        (self.latches.load(Ordering::SeqCst), last.clone())
    }

    /// Whether it is latched now.
    pub(crate) fn is_latched(&self) -> bool {
        self.latched.load(Ordering::SeqCst)
    }

    /// How many times it was latched.
    pub(crate) fn latches(&self) -> u64 {
        self.latches.load(Ordering::SeqCst)
    }

    /// The code of the last latch.
    pub(crate) fn code(&self) -> ReasonCode {
        lock(&self.reason).code
    }

    /// Why it is latched, while it is.
    pub(super) fn reason(&self) -> Option<Reason> {
        self.is_latched().then(|| lock(&self.reason).clone())
    }
}

/// What a robot process tells the host about its own stops, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Report {
    /// It fell back on its own account in `epoch`: the session goes SAFE.
    Safe {
        epoch: Epoch,
        code: ReasonCode,
        reason: String,
    },
    /// It latched an e-stop, its `seq`th: its own (`ESTOP_LOCAL`), or a
    /// member's signal (`ESTOP_SIGNAL`, from `origin`).
    Estop {
        reason: String,
        seq: u64,
        code: ReasonCode,
        origin: Option<ParticipantId>,
        /// When it latched, on this process's clock.
        at_ns: i64,
    },
    /// It cleared the latch, after `after` latches, the last for
    /// `reason`.
    Clear { after: u64, reason: Reason },
}

impl Report {
    /// The report of latch `seq`, for `reason`.
    pub(super) fn estop(reason: &Reason, seq: u64) -> Self {
        let origin = (reason.code == ReasonCode::EstopSignal)
            .then(|| ParticipantId::new(&reason.by).ok())
            .flatten();
        Self::Estop {
            reason: reason.detail.clone(),
            seq,
            code: reason.code,
            origin,
            at_ns: reason.at_ns,
        }
    }
}

/// Keeps a robot process's stop reports in order whatever order
/// its threads queued them in: a `Clear` goes out only after an `Estop` for
/// every latch before it, and an `Estop` a clear already covered goes
/// nowhere.
#[derive(Debug, Default)]
struct StopOrder {
    /// The latches an `Estop` went out for.
    estops: u64,
}

impl StopOrder {
    /// What `report` sends, in order.
    fn plan(&mut self, report: Report) -> Vec<Report> {
        match report {
            Report::Estop { seq, .. } if seq <= self.estops => Vec::new(),
            Report::Estop {
                reason,
                seq,
                code,
                origin,
                at_ns,
            } => {
                self.estops = seq;
                vec![Report::Estop {
                    reason,
                    seq,
                    code,
                    origin,
                    at_ns,
                }]
            }
            // The latch's own report hasn't gone out: the clear sends it,
            // with its reason.
            Report::Clear { after, reason } if after > self.estops => {
                self.estops = after;
                vec![
                    Report::estop(&reason, after),
                    Report::Clear { after, reason },
                ]
            }
            other => vec![other],
        }
    }
}

/// What the reporter does with one attempt's outcome.
#[derive(Debug, PartialEq, Eq)]
enum Attempt {
    /// The host answered (or refused it, authenticated): next report.
    Done,
    /// Send it again under the same request id.
    Retry,
    /// Its outcome is unknown (`EXPIRED`): send it again, anew.
    Anew,
}

impl Attempt {
    /// Judges `result`. A sealed refusal is the host's own and ends a
    /// report as an answer does (`EXPIRED` sends it anew). An unsealed one
    /// ([`SessionError::Unverified`]) is only a hint (anyone can answer
    /// `ctl`, and a superseded key gets one too): it never ends a report
    /// while this generation lasts (`ended`).
    fn of(result: &Result<(), SessionError>, ended: bool) -> Self {
        match result {
            Err(SessionError::Rejected(err)) if err.status == CtlStatus::Expired => Self::Anew,
            Ok(()) | Err(SessionError::Rejected(_)) => Self::Done,
            Err(SessionError::Unverified(_)) if ended => Self::Done,
            Err(_) => Self::Retry,
        }
    }
}

/// A handle that tells the host an edge fell back on its own account, for
/// code off the async path. Cheap to clone; see
/// [`Participant::report_safe`].
#[derive(Clone, Debug)]
pub struct SafeReporter {
    tx: mpsc::UnboundedSender<Report>,
}

impl SafeReporter {
    /// Queues an `EnterSafe` for a fault in `epoch`; never blocks.
    pub fn report(&self, epoch: Epoch, code: ReasonCode, reason: impl Into<String>) {
        // The reporter task is gone only after the participant closed.
        let _ = self.tx.send(Report::Safe {
            epoch,
            code,
            reason: reason.into(),
        });
    }
}

impl Participant {
    /// The report an edge on this participant updates; heartbeats carry it.
    pub fn edge_report(&self) -> Arc<EdgeReport> {
        self.shared.edge.clone()
    }

    /// Tells the host that this embodiment fell back on its own account
    /// (an edge's queue ran dry past `hold_max`, a lease lapsed by its
    /// clock, a device watchdog fired in `epoch`), so the session goes SAFE.
    /// Never blocks: a background task sends `EnterSafe`, retrying every
    /// 100 ms, until the host is SAFE or ESTOP at an epoch at or after
    /// `epoch`.
    pub fn report_safe(&self, epoch: Epoch, code: ReasonCode, reason: impl Into<String>) {
        self.safe_reporter().report(epoch, code, reason);
    }

    /// Latches this robot process's local e-stop: an atomic
    /// store, safe from any thread (a hardware button's handler too), that
    /// needs no network. Every edge of this process refuses chunks and runs
    /// Estop from its next tick; a background task reports
    /// `Estop{ESTOP_LOCAL}` to the host, retrying, in order with the other
    /// reports.
    pub fn latch_estop(&self, reason: impl Into<String>) {
        self.shared.latch(Reason {
            code: ReasonCode::EstopLocal,
            by: self.shared.id.to_string(),
            detail: reason.into(),
            at_ns: self.shared.clock.now_ns(),
            ..Reason::default()
        });
    }

    /// Why this robot process's e-stop is latched, while it is: what a
    /// process carries into its next session ([`ParticipantConfig::estop`]).
    pub fn local_estop(&self) -> Option<Reason> {
        self.shared.estop.reason()
    }

    /// Clears the local e-stop latch: the only way to, since no wire path
    /// clears it. `Clear` is queued behind any unsent `Estop`, so the host
    /// ends in ESTOP, unlatched, and leaving it still needs disarm and arm.
    /// It also clears a host latch this process did not set.
    pub fn clear_estop(&self) {
        let (after, reason) = self.shared.estop.clear();
        let _ = self.shared.reports.send(Report::Clear { after, reason });
    }

    /// Whether the local e-stop is latched.
    pub fn estop_latched(&self) -> bool {
        self.shared.estop.is_latched()
    }

    /// The latch, for this process's edges.
    pub(crate) fn estop_latch(&self) -> Arc<LocalLatch> {
        self.shared.estop.clone()
    }

    /// A handle for [`Participant::report_safe`] that outlives borrows.
    pub fn safe_reporter(&self) -> SafeReporter {
        SafeReporter {
            tx: self.shared.reports.clone(),
        }
    }
}

/// How often an unanswered stop report is resent.
pub(super) const SAFE_RETRY: Duration = Duration::from_millis(100);

/// Sends this robot process's stops to the host one at a time, in order,
/// each until the host answers (or refuses it). An `EnterSafe` also stops
/// once the view shows the host SAFE or ESTOP at an epoch at or after the
/// fault's (a re-armed host goes SAFE again: the fault may be a device's).
/// The edges have stopped already; this only tells the host.
pub(super) async fn report_stops(
    shared: Arc<Shared>,
    mut reports: mpsc::UnboundedReceiver<Report>,
    cancel: CancellationToken,
) {
    let mut order = StopOrder::default();
    loop {
        let report = tokio::select! {
            () = cancel.cancelled() => return,
            report = reports.recv() => match report {
                Some(report) => report,
                None => return,
            },
        };
        for report in order.plan(report) {
            if !send_stop(&shared, &report, &cancel).await {
                return;
            }
        }
    }
}

/// Sends one stop report until it is done ([`Attempt`]); `false` once the
/// participant closed.
async fn send_stop(shared: &Shared, report: &Report, cancel: &CancellationToken) -> bool {
    // One request id per report: a retry is answered from the host's reply
    // cache, never applied twice.
    let mut request_id = None;
    loop {
        let command = match &report {
            Report::Safe {
                epoch,
                code,
                reason,
            } => {
                let (now, mode) = shared.epoch_and_mode();
                if now >= epoch.0 && matches!(mode, Mode::Safe | Mode::Estop) {
                    return true;
                }
                Command::EnterSafe {
                    reason: reason.clone(),
                    code: *code,
                }
            }
            Report::Estop {
                reason,
                code,
                origin,
                at_ns,
                ..
            } => Command::Estop {
                reason: reason.clone(),
                code: *code,
                origin: origin.clone(),
                at_ns: *at_ns,
            },
            Report::Clear { .. } => Command::Clear,
        };
        let (id, reply) = shared.call(request_id, &command).await;
        request_id = id;
        let result = reply.and_then(|reply| reply.result.map_err(SessionError::Rejected));
        match Attempt::of(&result, shared.generation_ended()) {
            Attempt::Done => return true,
            Attempt::Anew => request_id = None,
            Attempt::Retry => {
                if let Err(err) = &result {
                    warn!(participant = %shared.id, %err, ?report, "stop report unanswered; retrying");
                }
                tokio::select! {
                    () = cancel.cancelled() => return false,
                    () = tokio::time::sleep(SAFE_RETRY) => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::CtlError;

    #[test]
    fn stop_reports_keep_estop_before_clear_and_retry_through_spoofed_refusals() {
        // The latch is a member's signal.
        let latch = Reason {
            code: ReasonCode::EstopSignal,
            by: "eye".into(),
            detail: "button".into(),
            ..Reason::default()
        };
        let estop = |seq| Report::estop(&latch, seq);
        let clear = |after| Report::Clear {
            after,
            reason: latch.clone(),
        };
        // A latch whose report was queued after the clear that followed
        // it: the clear sends its Estop first, with the latch's own
        // reason, and the late one nothing.
        let mut order = StopOrder::default();
        let sent: Vec<Report> = [clear(1), estop(1), estop(2), clear(2)]
            .into_iter()
            .flat_map(|r| order.plan(r))
            .collect();
        assert!(
            matches!(&sent[0], Report::Estop {
                reason,
                code: ReasonCode::EstopSignal,
                origin: Some(by),
                ..
            } if reason == "button" && by.as_str() == "eye"),
            "{sent:?}"
        );
        let kinds: Vec<&str> = sent
            .iter()
            .map(|r| match r {
                Report::Estop { .. } => "estop",
                Report::Clear { .. } => "clear",
                Report::Safe { .. } => "safe",
            })
            .collect();
        assert_eq!(kinds, ["estop", "clear", "estop", "clear"]);

        // A plain UNAUTHENTICATED (a MITM's, while the host's reply is
        // suppressed) keeps the Estop queued; the restored link's answer
        // ends it, and only then does the Clear go. The host's own sealed
        // UNAUTHENTICATED is an answer.
        let spoofed = Err(SessionError::Unverified(CtlError::new(
            CtlStatus::Unauthenticated,
            "spoofed",
        )));
        assert_eq!(Attempt::of(&spoofed, false), Attempt::Retry);
        assert_eq!(
            Attempt::of(&Err(SessionError::NoReply(Duration::ZERO)), false),
            Attempt::Retry
        );
        assert_eq!(Attempt::of(&Ok(()), false), Attempt::Done);
        // An authenticated refusal, an expired id, a generation that ended.
        let refused = |status| Err(SessionError::Rejected(CtlError::new(status, "")));
        for status in [CtlStatus::InvalidTransition, CtlStatus::Unauthenticated] {
            assert_eq!(Attempt::of(&refused(status), false), Attempt::Done);
        }
        assert_eq!(
            Attempt::of(&refused(CtlStatus::Expired), false),
            Attempt::Anew
        );
        assert_eq!(Attempt::of(&spoofed, true), Attempt::Done);
    }
}
