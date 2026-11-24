//! Per-op telemetry: the driver's op sources, recording RPC and peer-reported
//! timings into the aggregator, and the live window ticker.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use bytes::Bytes;

use super::PeerReport;
use crate::model::types::ModelObservation;
use crate::run::runtime::hooks::{RuntimeEnvContext, RuntimeHooks, TelemetrySnapshotEvent};
use crate::run::runtime::telemetry::{Aggregator, Horizon, Sample, Source, metrics};

pub(super) const SRC_PREDICT: Source = Source {
    op: "model.predict",
    component: "model",
};
pub(super) const SRC_RESET_ADAPTER: Source = Source {
    op: "model.reset_adapter",
    component: "model",
};
pub(super) const SRC_STEP: Source = Source {
    op: "env.step",
    component: "env",
};
pub(super) const SRC_RESET: Source = Source {
    op: "env.reset",
    component: "env",
};
pub(super) const SRC_TRANSFORM_OBS: Source = Source {
    op: "runner.transform_observation",
    component: "runner",
};
pub(super) const SRC_TRANSFORM_ACTION: Source = Source {
    op: "runner.transform_action",
    component: "runner",
};
pub(super) const SRC_ROUND: Source = Source {
    op: "runner.round",
    component: "runner",
};

pub(super) fn record_transform(telemetry: &Mutex<Aggregator>, source: Source, started: Instant) {
    lock_agg(telemetry).record(Sample::dur(source, metrics::RPC_TOTAL, started.elapsed()));
}

pub(super) fn leaf_bytes(leaves: &[Bytes]) -> u64 {
    leaves.iter().map(|leaf| leaf.len() as u64).sum()
}

pub(super) fn observation_bytes(observation: &ModelObservation) -> u64 {
    observation.observation.as_deref().map_or(0, leaf_bytes)
        + observation
            .history
            .iter()
            .map(|row| row.observation.as_deref().map_or(0, leaf_bytes))
            .sum::<u64>()
}

pub(super) fn lock_agg(telemetry: &Mutex<Aggregator>) -> MutexGuard<'_, Aggregator> {
    telemetry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(super) fn record_history_rows(telemetry: &Arc<Mutex<Aggregator>>, msg: &ModelObservation) {
    if !msg.history.is_empty() {
        lock_agg(telemetry).record(Sample::count(
            SRC_PREDICT,
            metrics::HISTORY_ROWS,
            msg.history.len() as u64,
        ));
    }
}

pub(super) fn record_op(
    telemetry: &Mutex<Aggregator>,
    src: Source,
    rpc: Duration,
    peer: PeerReport,
    request_bytes: u64,
    response_bytes: u64,
) {
    let mut agg = lock_agg(telemetry);
    agg.record(Sample::dur(src, metrics::RPC_TOTAL, rpc));
    if let Some(ns) = peer.endpoint_total_ns {
        agg.record(Sample::dur(
            src,
            metrics::ENDPOINT_TOTAL,
            Duration::from_nanos(ns),
        ));
    }
    for (metric, ns) in [
        (metrics::ENDPOINT_DECODE, peer.phases.decode_ns),
        (metrics::ENDPOINT_USER, peer.phases.user_ns),
        (metrics::ENDPOINT_ENCODE, peer.phases.encode_ns),
        (metrics::ENDPOINT_QUEUE, peer.phases.queue_ns),
        (metrics::PREDICT_ADAPTER, peer.phases.adapter_ns),
    ] {
        if ns != 0 {
            agg.record(Sample::dur(src, metric, Duration::from_nanos(ns)));
        }
    }
    if let Some(ns) = peer.phases.lane_skew_ns {
        agg.record(Sample::dur(
            src,
            metrics::LANE_SKEW,
            Duration::from_nanos(ns),
        ));
    }
    if peer.phases.in_flight != 0 {
        agg.record(Sample::count(
            src,
            metrics::PREDICT_IN_FLIGHT,
            u64::from(peer.phases.in_flight),
        ));
    }
    if let Some(episodes) = peer.phases.held_episodes {
        agg.record(Sample::count(
            src,
            metrics::HELD_EPISODES,
            u64::from(episodes),
        ));
    }
    if let Some(bytes) = peer.phases.held_state_bytes {
        agg.record(Sample::bytes(src, metrics::HELD_BYTES, bytes));
    }
    agg.record(Sample::bytes(src, metrics::REQUEST_BYTES, request_bytes));
    agg.record(Sample::bytes(src, metrics::RESPONSE_BYTES, response_bytes));
    if let Some(group) = peer.group_size {
        agg.record(Sample::count(src, metrics::GROUP_SIZE, group));
    }
}

pub(super) struct TelemetryTicker {
    handle: tokio::task::JoinHandle<()>,
}

impl TelemetryTicker {
    pub(super) fn spawn(
        telemetry: Arc<Mutex<Aggregator>>,
        hooks: Arc<dyn RuntimeHooks>,
        window: Duration,
        session_id: String,
        route: RuntimeEnvContext,
    ) -> Self {
        let period = window.max(Duration::from_millis(1));
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(period);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let window_snap = {
                    let mut agg = lock_agg(&telemetry);
                    let snap = agg.snapshot(Horizon::Window);
                    agg.flush_window();
                    snap
                };
                if window_snap.rows.is_empty() {
                    continue;
                }
                let window_event = TelemetrySnapshotEvent {
                    session_id: session_id.clone(),
                    route: route.clone(),
                    snapshot: window_snap,
                };
                if let Err(err) = hooks.on_telemetry(window_event).await {
                    tracing::warn!("runtime hook on_telemetry (window) failed: {err}");
                }
            }
        });
        Self { handle }
    }
}

impl Drop for TelemetryTicker {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
