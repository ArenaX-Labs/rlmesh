//! The batch sweep: how a served model's throughput grows with the number of
//! sessions predicting at once, and the `max_batch_size` it recommends.
//!
//! Ported from the managed probe's `batch_sweep`. The 0.1 probe sent one
//! grouped predict of N lanes; on 0.2 a served model's batcher groups the
//! predicts of concurrent sessions itself, so a sweep point is N routes
//! driven at once ([`run_routes`]), each a one-lane stand-in env, and its
//! latency the per-step predict time the routes see.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::model::{ModelProbeOptions, OBS_SEED_A};
use super::report::{BaselinePoint, CheckStatus, ProbeCheck, ProbeReport, ProbeSubject};
use super::synth::{Script, StandIn, SynthesizedEnv, serve_beside};
use super::{Probe, percentile};
use crate::binding::routes::{Route, merge_results, run_routes};
use crate::{RemoteModel, RemoteModelOptions, RunHooks, RunOptions, StepEvent, TelemetryRow};

const BENCH_CHECK_NAMES: &[&str] = &["handshake", "batch_sweep"];
/// The sweep's batch sizes: the platform's.
pub const SWEEP_SIZES: [u32; 6] = [1, 2, 4, 8, 16, 32];
const SWEEP_WARMUP: u32 = 3;
const SWEEP_ITERATIONS: u32 = 20;

/// What [`bench_model`] sweeps.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// The model's describe and reference env, as [`super::probe_model`]
    /// takes them.
    pub probe: ModelProbeOptions,
    /// Concurrent sessions per sweep point, ascending.
    pub sizes: Vec<u32>,
    /// Predicts each session times per point, after `warmup` untimed ones.
    pub iterations: u32,
    pub warmup: u32,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            probe: ModelProbeOptions::default(),
            sizes: SWEEP_SIZES.to_vec(),
            iterations: SWEEP_ITERATIONS,
            warmup: SWEEP_WARMUP,
        }
    }
}

/// A sweep: the report (its `handshake` and `batch_sweep` checks, the
/// baseline and the recommendation, in the probe report's shape) and the
/// telemetry of the single-session point, whose reset and step rows are the
/// round trips a run pays around each predict.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub report: ProbeReport,
    pub single: Vec<TelemetryRow>,
}

/// Sweep the model served at `address`.
pub fn bench_model(address: &str, options: &BenchOptions) -> BenchResult {
    let mut probe = Probe::new(ProbeSubject::Model, BENCH_CHECK_NAMES);
    let (synthesized, env, peer) = match super::model::connect(&mut probe, address, &options.probe)
    {
        Ok(connected) => connected,
        Err(peer) => {
            return BenchResult {
                report: probe.report(super::report::ProbeEnv::default(), *peer),
                single: Vec::new(),
            };
        }
    };
    let model = RemoteModel::new(
        RemoteModelOptions::default()
            .address(address)
            .connect_timeout(options.probe.connect_timeout)
            .request_timeout(options.probe.connect_timeout),
    );
    let single = match model {
        Ok(model) => sweep(&mut probe, &model, &synthesized, options),
        Err(error) => {
            probe.record(ProbeCheck::new(
                "batch_sweep",
                CheckStatus::Fail,
                error.to_string(),
            ));
            Vec::new()
        }
    };
    BenchResult {
        report: probe.report(env, peer),
        single,
    }
}

fn sweep(
    probe: &mut Probe,
    model: &RemoteModel,
    synthesized: &SynthesizedEnv,
    options: &BenchOptions,
) -> Vec<TelemetryRow> {
    let mut max_ok = None;
    let mut stop_reason = None;
    let mut single = Vec::new();
    for &n in &options.sizes {
        match point_retrying_a_dip(model, synthesized, n, options, probe.baseline.last()) {
            Ok((point, telemetry)) => {
                if n == 1 {
                    single = telemetry;
                }
                max_ok = Some(n);
                probe.baseline.push(point);
            }
            Err(error) => {
                stop_reason = Some((n, error));
                break;
            }
        }
    }
    probe.observed.max_ok_batch = max_ok;
    let mut check = ProbeCheck::new("batch_sweep", CheckStatus::Pass, String::new());
    match (max_ok, stop_reason) {
        (None, Some((n, error))) => {
            check.status = CheckStatus::Fail;
            check.detail = format!("N={n} failed: {error}");
        }
        (Some(max_ok), Some((n, error))) => {
            check.detail = format!("max ok batch {max_ok}; N={n} failed: {error}");
            check = check
                .tag("batch_limit")
                .metric("maxOkBatch", f64::from(max_ok));
        }
        (Some(max_ok), None) => {
            check.detail = format!("all sweep sizes up to {max_ok} succeeded");
            check = check.metric("maxOkBatch", f64::from(max_ok));
        }
        (None, None) => {
            check.status = CheckStatus::Fail;
            check.detail = "sweep ran no sizes".to_string();
        }
    }
    for point in &probe.baseline {
        check = check.metric(
            &format!("predictsPerSAtN{}", point.batch_size),
            point.predicts_per_s,
        );
    }
    probe.record(check);
    single
}

/// A size that measures slower than the one before it is run once more and
/// the better run kept: a first pass at a new shape is more often a warm-up
/// (JIT, allocator growth) than the knee.
fn point_retrying_a_dip(
    model: &RemoteModel,
    synthesized: &SynthesizedEnv,
    n: u32,
    options: &BenchOptions,
    previous: Option<&BaselinePoint>,
) -> Result<(BaselinePoint, Vec<TelemetryRow>), String> {
    let measured = point(model, synthesized, n, options)?;
    if !previous.is_some_and(|previous| measured.0.predicts_per_s < previous.predicts_per_s) {
        return Ok(measured);
    }
    match point(model, synthesized, n, options) {
        Ok(again) if again.0.predicts_per_s > measured.0.predicts_per_s => Ok(again),
        _ => Ok(measured),
    }
}

/// Times the predicts of the steps past `warmup`.
struct PredictTimes {
    warmup: i64,
    ms: Vec<f64>,
}

impl RunHooks for PredictTimes {
    fn on_step(&mut self, event: &StepEvent) -> crate::Result<()> {
        if event.step >= self.warmup {
            self.ms.push(event.predict_ms);
        }
        Ok(())
    }
}

/// One sweep point: `n` routes at once, each `warmup + iterations` steps.
fn point(
    model: &RemoteModel,
    synthesized: &SynthesizedEnv,
    n: u32,
    options: &BenchOptions,
) -> Result<(BaselinePoint, Vec<TelemetryRow>), String> {
    let servers = (0..n)
        .map(|_| {
            let script = Script {
                observations: vec![OBS_SEED_A],
                actions: Vec::new(),
            };
            serve_beside(
                model.address(),
                StandIn::new(synthesized, Arc::new(Mutex::new(script))),
            )
        })
        .collect::<crate::Result<Vec<_>>>()
        .map_err(|error| format!("serve the stand-in envs beside the model: {error}"))?;
    let mut hooks: Vec<PredictTimes> = (0..n)
        .map(|_| PredictTimes {
            warmup: i64::from(options.warmup),
            ms: Vec::new(),
        })
        .collect();
    let routes = servers
        .iter()
        .zip(hooks.iter_mut())
        .map(|(server, hooks)| {
            Route::new(
                server,
                RunOptions {
                    episodes: Some(1),
                    max_episode_steps: Some(u64::from(options.warmup + options.iterations)),
                    execution_horizon: Some(1),
                    hooks: Some(hooks),
                    ..Default::default()
                },
            )
        })
        .collect();
    let started = Instant::now();
    let results = run_routes(model, routes).map_err(|error| error.to_string());
    let elapsed = started.elapsed();
    for server in servers {
        let _ = server.shutdown();
    }
    let results = results?;
    let results = results
        .into_iter()
        .collect::<crate::Result<Vec<_>>>()
        .map_err(|error| error.to_string())?;
    let telemetry = merge_results(results.iter()).telemetry;
    let mut ms: Vec<f64> = hooks.into_iter().flat_map(|hooks| hooks.ms).collect();
    if ms.is_empty() {
        return Err(format!(
            "no predict timed within {:.1}s",
            Duration::as_secs_f64(&elapsed)
        ));
    }
    ms.sort_by(f64::total_cmp);
    let mean_ms = ms.iter().sum::<f64>() / ms.len() as f64;
    let point = BaselinePoint {
        batch_size: n,
        iterations: (ms.len() as u32) / n.max(1),
        p50_ms: percentile(&ms, 0.5),
        p95_ms: percentile(&ms, 0.95),
        endpoint_p50_ms: None,
        predicts_per_s: if mean_ms > 0.0 {
            f64::from(n) * 1e3 / mean_ms
        } else {
            0.0
        },
    };
    Ok((point, telemetry))
}
