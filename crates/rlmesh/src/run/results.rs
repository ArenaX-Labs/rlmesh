//! A run's results with Python's names and units (RUN-10): `RunResult`,
//! `EpisodeResult`, `TelemetryRow` and the per-step `StepEvent`.

use std::fmt;

use crate::env::factory::{py_float, py_repr, py_str};
use crate::{InfoDict, Value};

/// Python's `repr` of an optional float.
fn py_opt_float(value: Option<f64>) -> String {
    value.map_or_else(|| "None".to_string(), py_float)
}

/// Python's `repr` of an optional int.
fn py_opt_int(value: Option<i64>) -> String {
    value.map_or_else(|| "None".to_string(), |value| value.to_string())
}

/// Python's `repr` of a bool.
fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

/// Python's `f"{value:.{digits}f}"`.
fn py_fixed(value: f64, digits: usize) -> String {
    if value.is_nan() {
        "nan".to_string()
    } else {
        format!("{value:.digits$}")
    }
}

/// The outcome of one episode: Python's `EpisodeResult`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EpisodeResult {
    /// 0-based episode index within the run.
    pub index: i64,
    /// The seed the episode was reset with, or `None`.
    pub seed: Option<i64>,
    /// Env steps taken.
    pub steps: i64,
    /// Total reward over the episode.
    pub reward: f64,
    /// Whether the env reported a terminal state.
    pub terminated: bool,
    /// Whether the episode was cut short (env truncation, a cap, or the
    /// built-in step bound).
    pub truncated: bool,
    /// The env-reported task outcome from the final step's info, or `None`.
    pub success: Option<bool>,
    /// Wall time from reset-return to episode end, in seconds.
    pub duration_s: f64,
    /// The episode's mean per-step predict wall time, in milliseconds.
    pub predict_ms: Option<f64>,
    /// The episode's mean per-step env-step wall time, in milliseconds.
    pub step_ms: Option<f64>,
    /// The trial ordinal the episode walked, or `None`.
    pub trial: Option<i64>,
}

impl fmt::Display for EpisodeResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EpisodeResult(index={}, seed={}, steps={}, reward={}, terminated={}, truncated={}, \
             success={}, duration_s={}, predict_ms={}, step_ms={}, trial={})",
            self.index,
            py_opt_int(self.seed),
            self.steps,
            py_float(self.reward),
            py_bool(self.terminated),
            py_bool(self.truncated),
            self.success.map_or("None", py_bool),
            py_float(self.duration_s),
            py_opt_float(self.predict_ms),
            py_opt_float(self.step_ms),
            py_opt_int(self.trial),
        )
    }
}

/// One aggregated metric series of a run: Python's `TelemetryRow`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TelemetryRow {
    /// The measured operation (`model.predict`, `env.step`, ...).
    pub op: String,
    /// Who produced the sample (`model`, `env`, `runner`).
    pub component: String,
    /// The metric name (`rpc.total`, `endpoint.user`, ...).
    pub metric: String,
    /// What the values measure: `ms`, `bytes` or `count`.
    pub unit: String,
    /// Samples aggregated into this row.
    pub count: i64,
    /// Mean value, in `unit`.
    pub avg: f64,
    /// Median value, in `unit`.
    pub p50: f64,
    /// 95th-percentile value, in `unit`.
    pub p95: f64,
    /// 99th-percentile value, in `unit`.
    pub p99: f64,
}

impl fmt::Display for TelemetryRow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TelemetryRow(op={}, component={}, metric={}, unit={}, count={}, avg={}, p50={}, \
             p95={}, p99={})",
            py_str(&self.op),
            py_str(&self.component),
            py_str(&self.metric),
            py_str(&self.unit),
            self.count,
            py_float(self.avg),
            py_float(self.p50),
            py_float(self.p95),
            py_float(self.p99),
        )
    }
}

/// The result of a run: Python's `RunResult`.
///
/// Python's `advisories` field arrives with the adapters surface's
/// `Advisory`; the open-source runtime never raises one.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RunResult {
    /// Every completed episode.
    pub episodes: Vec<EpisodeResult>,
    /// The run's aggregated telemetry, one row per measured series.
    pub telemetry: Vec<TelemetryRow>,
}

impl RunResult {
    /// Number of episodes in this result.
    pub fn num_episodes(&self) -> usize {
        self.episodes.len()
    }

    /// Total env steps across all episodes.
    pub fn total_steps(&self) -> i64 {
        self.episodes.iter().map(|episode| episode.steps).sum()
    }

    /// Mean total reward per episode (`0.0` when empty).
    pub fn mean_reward(&self) -> f64 {
        if self.episodes.is_empty() {
            return 0.0;
        }
        self.episodes
            .iter()
            .map(|episode| episode.reward)
            .sum::<f64>()
            / self.episodes.len() as f64
    }

    /// The fraction of episodes the env reported a success, or `None` when
    /// the run is empty or any episode lacks the signal.
    pub fn success_rate(&self) -> Option<f64> {
        if self.episodes.is_empty() || self.episodes.iter().any(|e| e.success.is_none()) {
            return None;
        }
        let successes = self
            .episodes
            .iter()
            .filter(|episode| episode.success == Some(true))
            .count();
        Some(successes as f64 / self.episodes.len() as f64)
    }

    /// The telemetry rows as an aligned text table: Python's
    /// `format_telemetry()`.
    pub fn format_telemetry(&self) -> String {
        if self.telemetry.is_empty() {
            return "(no telemetry recorded)".to_string();
        }
        let header = ["op", "metric", "unit", "count", "avg", "p50", "p95", "p99"];
        let rows: Vec<[String; 8]> = self
            .telemetry
            .iter()
            .map(|row| {
                [
                    row.op.clone(),
                    row.metric.clone(),
                    row.unit.clone(),
                    row.count.to_string(),
                    py_fixed(row.avg, 2),
                    py_fixed(row.p50, 2),
                    py_fixed(row.p95, 2),
                    py_fixed(row.p99, 2),
                ]
            })
            .collect();
        let widths: Vec<usize> = header
            .iter()
            .enumerate()
            .map(|(i, title)| {
                rows.iter()
                    .map(|row| row[i].chars().count())
                    .chain([title.chars().count()])
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let line = |cells: &[String]| {
            cells
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let pad = widths[i].saturating_sub(cell.chars().count());
                    if i < 3 {
                        format!("{cell}{}", " ".repeat(pad))
                    } else {
                        format!("{}{cell}", " ".repeat(pad))
                    }
                })
                .collect::<Vec<_>>()
                .join("  ")
        };
        let header: Vec<String> = header.iter().map(|title| title.to_string()).collect();
        std::iter::once(line(&header))
            .chain(rows.iter().map(|row| line(row)))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RunResult(episodes={}, mean_reward={}, total_steps={})",
            self.num_episodes(),
            py_fixed(self.mean_reward(), 3),
            self.total_steps()
        )
    }
}

/// One env step of a run, passed to
/// [`RunHooks::on_step`](crate::RunHooks::on_step): Python's `StepEvent`.
///
/// Python's `read` member arrives with the adapters surface's read items.
#[derive(Clone, Debug, PartialEq)]
pub struct StepEvent {
    /// 0-based episode index (equals [`EpisodeResult::index`]).
    pub episode: i64,
    /// The episode's reset seed, or `None`.
    pub seed: Option<i64>,
    /// 0-based step index within the episode.
    pub step: i64,
    /// The observation the action was predicted from (pre-step).
    pub observation: Value,
    /// The env-ready action applied to the env (never absent, RUN-21).
    pub action: Value,
    /// The step's reward.
    pub reward: f64,
    /// Whether the env reported a terminal state on this step.
    pub terminated: bool,
    /// Whether this step truncated the episode.
    pub truncated: bool,
    /// The step's info.
    pub info: InfoDict,
    /// Wall time of the predict(s) charged to this step, in milliseconds.
    pub predict_ms: f64,
    /// Wall time of the env step round trip, in milliseconds.
    pub step_ms: f64,
}

impl fmt::Display for StepEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StepEvent(episode={}, seed={}, step={}, observation={}, action={}, reward={}, \
             terminated={}, truncated={}, info={}, predict_ms={}, step_ms={})",
            self.episode,
            py_opt_int(self.seed),
            self.step,
            py_repr(&self.observation),
            py_repr(&self.action),
            py_float(self.reward),
            py_bool(self.terminated),
            py_bool(self.truncated),
            py_repr(&Value::Dict(self.info.clone())),
            py_float(self.predict_ms),
            py_float(self.step_ms),
        )
    }
}

/// The facade result of a native run's report (RUN-10: 0-based index,
/// seconds, Python's field names).
pub(crate) fn from_report(report: &rlmesh_runtime::RuntimeReport) -> RunResult {
    RunResult {
        episodes: report.episodes.iter().map(episode_from_summary).collect(),
        telemetry: telemetry_rows(&report.telemetry),
    }
}

pub(crate) fn episode_from_summary(
    episode: &rlmesh_runtime::spec::EpisodeSummary,
) -> EpisodeResult {
    EpisodeResult {
        index: episode.episode_index - 1,
        seed: episode.seed,
        steps: episode.step_count,
        reward: episode.cumulative_reward,
        terminated: episode.terminated,
        truncated: episode.truncated,
        success: episode.success,
        duration_s: episode.duration_ms as f64 / 1000.0,
        predict_ms: episode.predict_ms,
        step_ms: episode.step_ms,
        trial: episode.trial_index.map(|trial| trial as i64),
    }
}

/// One row per `(op, component, metric)` of a telemetry snapshot.
pub(crate) fn telemetry_rows(snapshot: &rlmesh_runtime::telemetry::Snapshot) -> Vec<TelemetryRow> {
    use rlmesh_runtime::telemetry::Kind;
    snapshot
        .rows
        .iter()
        .map(|row| TelemetryRow {
            op: row.source.op.to_string(),
            component: row.source.component.to_string(),
            metric: row.metric.name.to_string(),
            unit: match row.metric.kind {
                Kind::Duration => "ms",
                Kind::Bytes => "bytes",
                Kind::Count => "count",
            }
            .to_string(),
            count: row.count as i64,
            avg: row.avg,
            p50: row.p50,
            p95: row.p95,
            p99: row.p99,
        })
        .collect()
}
