//! Telemetry — windowed/session aggregation of per-op metrics.
//!
//! One ingest path ([`Aggregator::record`]), one shape out ([`Snapshot`]). A
//! metric is identified by its `&'static str` name in the [`metrics`] catalog;
//! window and session are retention horizons over the same Vitter reservoirs.
//! The per-step hot scalar `endpoint_total_ns` is the *wire* contract, not part
//! of this system — it is recorded here like any other Duration sample
//! (`metrics::ENDPOINT_TOTAL`).

mod reservoir;
mod stats;

use reservoir::ValueReservoir;
use std::collections::BTreeMap;
use std::time::Duration;

/// What a metric measures — fixes its unit (ms for `Duration`, bytes for `Bytes`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Duration,
    Bytes,
}

/// A metric's identity (`name`) + `kind`. Defined only in the [`metrics`] catalog.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metric {
    pub name: &'static str,
    pub kind: Kind,
}

impl Metric {
    pub const fn duration(name: &'static str) -> Self {
        Self {
            name,
            kind: Kind::Duration,
        }
    }
    pub const fn bytes(name: &'static str) -> Self {
        Self {
            name,
            kind: Kind::Bytes,
        }
    }
}

/// The metric catalog — adding a metric is one const here (plus `ALL`).
pub mod metrics {
    use super::Metric;

    pub const ENDPOINT_TOTAL: Metric = Metric::duration("endpoint.total");
    pub const RPC_TOTAL: Metric = Metric::duration("rpc.total");
    pub const REQUEST_BYTES: Metric = Metric::bytes("request.bytes");
    pub const RESPONSE_BYTES: Metric = Metric::bytes("response.bytes");

    /// The cardinality allowlist — derived from the catalog, not a second table.
    pub const ALL: &[Metric] = &[ENDPOINT_TOTAL, RPC_TOTAL, REQUEST_BYTES, RESPONSE_BYTES];
}

/// Who produced a sample: an `operation` on a `component`.
// ponytail: &'static str — the serial single-route driver has a fixed small set
// of (op, component) pairs. Widen to Cow<'static, str> if component_id goes dynamic.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Source {
    pub op: &'static str,
    pub component: &'static str,
}

/// One observation — the only thing [`Aggregator::record`] ingests.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub source: Source,
    pub metric: Metric,
    /// ms for a `Duration` metric; raw byte count for a `Bytes` metric.
    pub value: f64,
}

impl Sample {
    pub fn dur(source: Source, metric: Metric, d: Duration) -> Self {
        Self {
            source,
            metric,
            value: d.as_secs_f64() * 1e3,
        }
    }
    pub fn bytes(source: Source, metric: Metric, n: u64) -> Self {
        Self {
            source,
            metric,
            value: n as f64,
        }
    }
}

/// Retention horizon over the same recorded samples.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Horizon {
    /// Cleared each `flush_window`.
    Window,
    /// Whole session.
    Session,
}

/// One aggregated metric row. `avg`/`p*` are in the metric's unit (ms or bytes).
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: Source,
    pub metric: Metric,
    pub count: u64,
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

/// A point-in-time aggregate for one horizon — the one shape every consumer sees.
#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub horizon: Horizon,
    pub rows: Vec<Row>,
}

struct Series {
    source: Source,
    metric: Metric,
    window: ValueReservoir,
    session: ValueReservoir,
}

impl Series {
    fn new(source: Source, metric: Metric) -> Self {
        Self {
            source,
            metric,
            window: ValueReservoir::default(),
            session: ValueReservoir::default(),
        }
    }

    fn summarize(&self, horizon: Horizon) -> Option<Row> {
        let reservoir = match horizon {
            Horizon::Window => &self.window,
            Horizon::Session => &self.session,
        };
        let stats = stats::summary(reservoir.samples())?;
        Some(Row {
            source: self.source,
            metric: self.metric,
            count: reservoir.seen(),
            avg: stats.avg,
            p50: stats.p50,
            p95: stats.p95,
            p99: stats.p99,
        })
    }
}

/// Windowed / session metric aggregator. One `record` in, `snapshot` out.
#[derive(Default)]
pub struct Aggregator {
    series: BTreeMap<(&'static str, &'static str, &'static str), Series>,
}

impl Aggregator {
    /// The one ingest path. Every measurement in the system goes through here.
    pub fn record(&mut self, sample: Sample) {
        debug_assert!(
            metrics::ALL.iter().any(|m| m.name == sample.metric.name),
            "unregistered metric `{}` (add it to telemetry::metrics)",
            sample.metric.name,
        );
        let key = (
            sample.source.op,
            sample.source.component,
            sample.metric.name,
        );
        let series = self
            .series
            .entry(key)
            .or_insert_with(|| Series::new(sample.source, sample.metric));
        series.window.push(sample.value);
        series.session.push(sample.value);
    }

    /// Aggregate one horizon into a snapshot. Series with no samples this horizon
    /// are omitted (e.g. every series right after `flush_window` for `Window`).
    pub fn snapshot(&self, horizon: Horizon) -> Snapshot {
        Snapshot {
            horizon,
            rows: self
                .series
                .values()
                .filter_map(|s| s.summarize(horizon))
                .collect(),
        }
    }

    /// Clear the window horizon — call after emitting a window snapshot. Reuses
    /// each reservoir's backing capacity (no per-flush realloc).
    pub fn flush_window(&mut self) {
        for s in self.series.values_mut() {
            s.window.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src() -> Source {
        Source {
            op: "model.predict",
            component: "model",
        }
    }

    fn row<'a>(snap: &'a Snapshot, name: &str) -> &'a Row {
        snap.rows
            .iter()
            .find(|r| r.metric.name == name)
            .expect("row present")
    }

    #[test]
    fn aggregates_percentiles_and_flushes_window() {
        let mut agg = Aggregator::default();
        for ms in [10u64, 20, 30, 40] {
            agg.record(Sample::dur(
                src(),
                metrics::RPC_TOTAL,
                Duration::from_millis(ms),
            ));
        }

        // Window snapshot: count + ordered percentiles, durations reported in ms.
        let w = agg.snapshot(Horizon::Window);
        let r = row(&w, "rpc.total");
        assert_eq!(r.count, 4);
        assert!((r.avg - 25.0).abs() < 1e-9);
        assert!(r.p50 <= r.p95 && r.p95 <= r.p99);
        assert!((r.p99 - 40.0).abs() < 1e-9);

        // flush clears the window, but session retains.
        agg.flush_window();
        assert!(agg.snapshot(Horizon::Window).rows.is_empty());
        assert_eq!(row(&agg.snapshot(Horizon::Session), "rpc.total").count, 4);
    }
}
