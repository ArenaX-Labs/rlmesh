//! The model probe: handshake and describe, resolution against a stand-in env,
//! action conformance, batched rows, determinism, seeding and statefulness.
//!
//! Ported from the managed probe's model checks. A model is bound with the
//! 0.2 binder to a [`StandIn`] env whose observations the probe scripts, so
//! every check compares the model's answers to inputs it chose. The 0.1-wire
//! legs (`chunk_horizon`, `native_chunk_contract`, `grouped`,
//! `batched_seeded_repro`) have no 0.2 counterpart: the binder owns chunking,
//! and a served model's batcher groups concurrent sessions on its own side.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rlmesh_spaces::SpaceValue;
use serde_json::Value as Json;

use super::compare::{
    Drift, Frames, action_range, frames_max_abs_diff, range_fraction_note, replay_drift,
    shape_check, stateful_check, trials_spread,
};
use super::report::{CheckStatus, ProbeCheck, ProbeEnv, ProbePeer, ProbeReport, ProbeSubject};
use super::synth::{self, Script, StandIn, SynthesizedEnv};
use super::{DEFAULT_CONNECT_TIMEOUT, Probe, peer_from_describe};
use crate::binding::routes::{Route, run_routes};
use crate::model::ModelHandler as _;
use crate::{BindOptions, Binding, RemoteModel, RemoteModelOptions, RunOptions};

pub(super) const MODEL_CHECK_NAMES: &[&str] = &[
    "handshake",
    "resolve",
    "shape_single",
    "shape_batch",
    "determinism_floor",
    "seeded_repro",
    "stateless",
];
pub(super) const OBS_SEED_A: u64 = 0x5EED_000A;
const OBS_SEED_B: u64 = 0x5EED_000B;
/// Every probe episode is seeded, as an evaluation's episodes are: a model
/// that seeds itself from the episode seed must exercise that path here.
const EPISODE_SEED: i64 = 0x5EED_0E00;
/// Unseeded episodes on one input; their spread is the determinism floor.
const DETERMINISM_TRIALS: usize = 4;
/// Episodes all on the same seed and observation: a model that seeds itself
/// answers all of them identically however much it samples.
const REPRO_TRIALS: usize = 4;
const REPRO_EPISODE_SEED: i64 = 0x5EED_0E11;
/// Two further seeds for the sensitivity leg of `seeded_repro`: a sampling
/// model must answer them differently, or its seed never reaches the sampler.
const SENSITIVITY_SEEDS: [i64; 2] = [0x5EED_0E22, 0x5EED_0E33];
/// Concurrent sessions `shape_batch` joins the served model from, so its
/// batcher hands them to one forward.
const BATCH_ROWS: usize = 2;
const STATE_TOLERANCE: f64 = 8.0;
const STATE_ATOL: f64 = 1e-6;

/// What [`probe_model`] needs besides the model's address.
#[derive(Debug, Clone)]
pub struct ModelProbeOptions {
    /// The image's describe envelope (its `dev.rlmesh.describe` label); `None`
    /// takes the one the served model answers with.
    pub describe: Option<Json>,
    /// A served env whose live contract an adapter-less model (one that
    /// declares no `ModelSpec`) is checked against; the probe samples its
    /// spaces and never steps it.
    pub reference_env: Option<String>,
    /// How long the model gets to answer the handshake and each call.
    pub connect_timeout: Duration,
}

impl Default for ModelProbeOptions {
    fn default() -> Self {
        Self {
            describe: None,
            reference_env: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Probe the model served at the session participant `address`. Never fails:
/// a check that cannot run is recorded as failed or skipped in the report.
pub fn probe_model(address: &str, options: &ModelProbeOptions) -> ProbeReport {
    let mut probe = Probe::new(ProbeSubject::Model, MODEL_CHECK_NAMES);
    let (stand_in, env, peer) = match connect(&mut probe, address, options) {
        Ok(connected) => connected,
        Err(peer) => return probe.report(ProbeEnv::default(), *peer),
    };
    probe.observed.render_requests = stand_in.render_requests.clone();
    if let Some(chunk) = peer
        .describe
        .as_ref()
        .and_then(|envelope| envelope.get("native_chunk"))
        .and_then(Json::as_u64)
    {
        probe.observed.native_chunk = u32::try_from(chunk).ok();
        probe.observed.native_chunk_declared = Some(true);
    }
    let model = match RemoteModel::new(
        RemoteModelOptions::default()
            .address(address)
            .connect_timeout(options.connect_timeout)
            .request_timeout(options.connect_timeout),
    ) {
        Ok(model) => model,
        Err(error) => {
            probe.record(ProbeCheck::new(
                "resolve",
                CheckStatus::Fail,
                error.to_string(),
            ));
            probe.skip_remaining("resolve failed");
            return probe.report(env, peer);
        }
    };
    run_checks(&mut probe, &model, &stand_in);
    probe.report(env, peer)
}

/// The handshake: the model's describe over its session, then the stand-in
/// its inputs are synthesized into. On failure the report's peer.
pub(super) fn connect(
    probe: &mut Probe,
    address: &str,
    options: &ModelProbeOptions,
) -> Result<(SynthesizedEnv, ProbeEnv, ProbePeer), Box<ProbePeer>> {
    let started = Instant::now();
    let described = crate::run::drive::block_on_library(async {
        let mut served = crate::lockstep::ServedModel::new(address, options.connect_timeout);
        let described = served.describe().await;
        let _ = served.on_close().await;
        described
    });
    let served_describe = match described {
        Ok(json) => serde_json::from_str::<Json>(&json).ok(),
        Err(error) => {
            probe.record(
                ProbeCheck::new("handshake", CheckStatus::Fail, error.to_string())
                    .tag("connect_failed"),
            );
            probe.skip_remaining("handshake failed");
            return Err(Box::default());
        }
    };
    let connect_ms = started.elapsed().as_secs_f64() * 1e3;
    let describe = options.describe.clone().or(served_describe);
    let peer = peer_from_describe(describe.as_ref());
    let model_spec = describe
        .as_ref()
        .and_then(|envelope| envelope.get("model_spec"))
        .cloned()
        .unwrap_or(Json::Null);
    // A model that described itself but publishes no ModelSpec takes its
    // inputs from whichever env it is paired with, so there is nothing to
    // synthesize a stand-in from: with a reference env it is checked against
    // that env's live spaces; without one the handshake is all the probe can
    // verify. A missing describe still fails at synthesis below.
    let adapter_less = model_spec.is_null() && describe.is_some();
    let synthesized = match (&options.reference_env, adapter_less) {
        (Some(reference), true) => {
            reference_env(reference, options.connect_timeout).map_err(|error| {
                (
                    ProbeCheck::new(
                        "handshake",
                        CheckStatus::Fail,
                        format!("reference env: {error}"),
                    )
                    .tag("connect_failed"),
                    "the reference env did not connect",
                )
            })
        }
        (None, true) => {
            probe.record(
                ProbeCheck::new(
                    "handshake",
                    CheckStatus::Warn,
                    format!(
                        "model {} on {} {} declares no ModelSpec (adapter-less) and no reference \
                         env was given; its inputs are checked when an evaluation binds it",
                        peer.package_version, peer.language, peer.language_version
                    ),
                )
                .tag("adapter_less"),
            );
            probe.skip_remaining("the model declares no ModelSpec and no reference env was given");
            return Err(Box::new(peer));
        }
        (_, false) => synth::synthesize(&model_spec).map_err(|error| {
            (
                ProbeCheck::new("handshake", CheckStatus::Fail, error).tag("no_model_spec"),
                if describe.is_none() {
                    "the model's inputs could not be synthesized: it answered with no describe \
                     envelope and the image carries no label"
                } else {
                    "the model's inputs could not be synthesized"
                },
            )
        }),
    };
    let synthesized = match synthesized {
        Ok(synthesized) => synthesized,
        Err((check, reason)) => {
            probe.record(check);
            probe.skip_remaining(reason);
            return Err(Box::new(peer));
        }
    };
    probe.record(
        ProbeCheck::new(
            "handshake",
            CheckStatus::Pass,
            format!(
                "model {address} rlmesh {} on {} {}",
                peer.package_version, peer.language, peer.language_version
            ),
        )
        .metric("connectMs", connect_ms),
    );
    // The report names the env the model was checked against, so a reader
    // can tell a reference env's numbers from a synthesized one's.
    let env = ProbeEnv {
        repo: String::new(),
        env_id: match &options.reference_env {
            Some(reference) if adapter_less => reference.clone(),
            _ => synth::SYNTHETIC_ENV_ID.to_string(),
        },
        digest: String::new(),
    };
    Ok((synthesized, env, peer))
}

/// The spaces an adapter-less model is checked against: its reference env's
/// live contract, in place of a stand-in synthesized from a ModelSpec.
fn reference_env(address: &str, timeout: Duration) -> Result<SynthesizedEnv, String> {
    let mut binding = super::bind_env(address, timeout).map_err(|error| error.to_string())?;
    let contract = binding.contract().clone();
    let _ = binding.close();
    let tags = contract.to_wire().ok().and_then(|wire| {
        wire.metadata?
            .get(rlmesh_adapters::v1::ENV_METADATA_KEY)
            .map(crate::lockstep::meta_json::value)
    });
    Ok(SynthesizedEnv {
        observation: contract.observation_space().wire().clone(),
        action: contract.action_space().wire().clone(),
        tags,
        unsupported: Vec::new(),
        render_requests: Vec::new(),
    })
}

fn run_checks(probe: &mut Probe, model: &RemoteModel, synthesized: &SynthesizedEnv) {
    let script = Arc::new(Mutex::new(Script::default()));
    let started = Instant::now();
    let server = match synth::serve_beside(
        model.address(),
        StandIn::new(synthesized, Arc::clone(&script)),
    ) {
        Ok(server) => server,
        Err(error) => {
            probe.record(ProbeCheck::new(
                "resolve",
                CheckStatus::Fail,
                format!("serve the stand-in env beside the model: {error}"),
            ));
            probe.skip_remaining("resolve failed");
            return;
        }
    };
    // Re-plan every step, so each step is one predict on the input the
    // script served.
    let bound = crate::bind(
        model,
        &server,
        BindOptions {
            execution_horizon: Some(1),
            ..Default::default()
        },
    );
    let mut binding = match bound {
        Ok(binding) => binding,
        Err(error) => {
            probe.record(ProbeCheck::new(
                "resolve",
                CheckStatus::Fail,
                error.to_string(),
            ));
            probe.skip_remaining("resolve failed");
            let _ = server.shutdown();
            return;
        }
    };
    // Inputs with no declared role were left out of the stand-in env, so
    // the resolve that succeeds here covered less than the model asks for.
    let unsupported = if synthesized.unsupported.is_empty() {
        String::new()
    } else {
        format!(
            "; {} model input(s) could not be synthesized: {}",
            synthesized.unsupported.len(),
            synthesized.unsupported.join(", ")
        )
    };
    probe.record(
        ProbeCheck::new(
            "resolve",
            if unsupported.is_empty() {
                CheckStatus::Pass
            } else {
                CheckStatus::Warn
            },
            format!(
                "bound to {} at execution_horizon=1{unsupported}",
                binding.contract().id()
            ),
        )
        .metric("resolveMs", started.elapsed().as_secs_f64() * 1e3),
    );
    let mut checks = ModelChecks {
        probe,
        binding: &mut binding,
        script: &script,
        action: synthesized.action.clone(),
    };
    if checks.check_shape_single() {
        checks.check_shape_batch(model, synthesized);
        let floor = checks.check_determinism();
        checks.check_seeded_repro(floor);
        checks.check_stateless(floor);
    } else {
        checks.probe.skip_remaining("shape_single failed");
    }
    let _ = binding.close();
    let _ = server.shutdown();
}

struct ModelChecks<'p, 'b> {
    probe: &'p mut Probe,
    binding: &'p mut Binding<'b>,
    script: &'p Arc<Mutex<Script>>,
    action: rlmesh_spaces::SpaceSpec,
}

impl ModelChecks<'_, '_> {
    /// One episode reset with `seed`, served the observations `script` names
    /// (the reset's first, then one per step), stepped once per input: the
    /// model's action for each, as one-frame, one-lane predicts.
    fn episode(&mut self, seed: Option<i64>, script: &[u64]) -> Result<Vec<Frames>, String> {
        self.script
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .observations = script.to_vec();
        self.binding
            .reset(seed, None)
            .map_err(|error| format!("reset: {error}"))?;
        for _ in script {
            self.binding
                .step(None)
                .map_err(|error| format!("predict: {error}"))?;
        }
        let actions = std::mem::take(
            &mut self
                .script
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .actions,
        );
        if actions.len() != script.len() {
            return Err(format!(
                "{} step(s) delivered {} action(s) to the env",
                script.len(),
                actions.len()
            ));
        }
        Ok(actions
            .into_iter()
            .map(|action| vec![vec![action]])
            .collect())
    }

    /// The first action of `trials` episodes on input A, each reset with the
    /// seed `seeds` yields.
    fn first_actions(
        &mut self,
        seeds: impl IntoIterator<Item = Option<i64>>,
    ) -> Result<Vec<Frames>, String> {
        seeds
            .into_iter()
            .map(|seed| Ok(self.episode(seed, &[OBS_SEED_A])?.remove(0)))
            .collect()
    }

    fn check_shape_single(&mut self) -> bool {
        let check = match self.episode(Some(EPISODE_SEED), &[OBS_SEED_A]) {
            Ok(mut frames) => shape_check("shape_single", &frames.remove(0), 1, &self.action),
            Err(error) => ProbeCheck::new("shape_single", CheckStatus::Fail, error),
        };
        let ok = check.status != CheckStatus::Fail;
        self.probe.record(check);
        ok
    }

    /// `BATCH_ROWS` sessions predicting at once, which the served model's
    /// batcher may hand to one forward: each answer must still fit the
    /// action space.
    fn check_shape_batch(&mut self, model: &RemoteModel, synthesized: &SynthesizedEnv) {
        let scripts: Vec<Arc<Mutex<Script>>> = (0..BATCH_ROWS)
            .map(|_| {
                Arc::new(Mutex::new(Script {
                    observations: vec![OBS_SEED_A],
                    actions: Vec::new(),
                }))
            })
            .collect();
        let servers = scripts
            .iter()
            .map(|script| {
                synth::serve_beside(
                    model.address(),
                    StandIn::new(synthesized, Arc::clone(script)),
                )
            })
            .collect::<crate::Result<Vec<_>>>();
        let servers = match servers {
            Ok(servers) => servers,
            Err(error) => {
                self.probe.record(ProbeCheck::new(
                    "shape_batch",
                    CheckStatus::Fail,
                    format!("serve the stand-in envs beside the model: {error}"),
                ));
                return;
            }
        };
        let routes = servers
            .iter()
            .map(|server| {
                Route::new(
                    server,
                    RunOptions {
                        episodes: Some(1),
                        max_episode_steps: Some(2),
                        execution_horizon: Some(1),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let failure = match run_routes(model, routes) {
            Ok(results) => results
                .into_iter()
                .find_map(Result::err)
                .map(|error| error.to_string()),
            Err(error) => Some(error.to_string()),
        };
        for server in servers {
            let _ = server.shutdown();
        }
        let check = match failure {
            Some(error) => ProbeCheck::new(
                "shape_batch",
                CheckStatus::Warn,
                format!("{BATCH_ROWS} concurrent sessions failed: {error}"),
            )
            .tag("no_row_batch"),
            None => {
                let rows: Vec<Vec<SpaceValue>> = scripts
                    .iter()
                    .map(|script| {
                        std::mem::take(
                            &mut script.lock().unwrap_or_else(|e| e.into_inner()).actions,
                        )
                    })
                    .collect();
                let steps = rows.iter().map(Vec::len).min().unwrap_or(0);
                let frames: Frames = (0..steps)
                    .map(|step| rows.iter().map(|row| row[step].clone()).collect())
                    .collect();
                shape_check("shape_batch", &frames, BATCH_ROWS, &self.action)
            }
        };
        self.probe.record(check);
    }

    /// Unseeded episodes on one input: a model that samples answers them
    /// differently, and how far apart is the floor every later spread is
    /// read against. (The 0.1 probe repeated one seeded lane instead; a model
    /// that seeds itself read as deterministic there.)
    fn check_determinism(&mut self) -> Option<f64> {
        let name = "determinism_floor";
        let range = action_range(&self.action);
        let (check, floor) = match self.first_actions((0..DETERMINISM_TRIALS).map(|_| None)) {
            Err(error) => (ProbeCheck::new(name, CheckStatus::Fail, error), None),
            Ok(trials) => match trials_spread(&trials) {
                Some(spread) => {
                    let (status, detail) = if spread.max == 0.0 {
                        (
                            CheckStatus::Pass,
                            format!(
                                "identical actions across {DETERMINISM_TRIALS} unseeded episodes"
                            ),
                        )
                    } else {
                        (
                            CheckStatus::Warn,
                            format!(
                                "{DETERMINISM_TRIALS} unseeded episodes differ by max {:.3e}{} (mean {:.3e}, {:.1}% of {} action elements)",
                                spread.max,
                                range_fraction_note(spread.max, range),
                                spread.mean(),
                                spread.differing_fraction() * 100.0,
                                spread.elements
                            ),
                        )
                    };
                    let mut check = ProbeCheck::new(name, status, detail)
                        .metric("trials", DETERMINISM_TRIALS as f64)
                        .metric("maxAbsDiff", spread.max)
                        .metric("meanAbsDiff", spread.mean())
                        .metric("differingFraction", spread.differing_fraction())
                        .metric("elements", spread.elements as f64);
                    if let Some(range) = range {
                        check = check
                            .metric("actionRange", range)
                            .metric("maxRangeFraction", spread.max / range);
                    }
                    if spread.max != 0.0 {
                        check = check.tag("nondeterministic");
                    }
                    (check, Some(spread.max))
                }
                None => (
                    ProbeCheck::new(
                        name,
                        CheckStatus::Fail,
                        "unseeded episodes returned incomparable action shapes",
                    )
                    .tag("nondeterministic_shape"),
                    None,
                ),
            },
        };
        self.probe.observed.deterministic = floor.map(|floor| floor == 0.0);
        self.probe.record(check);
        floor
    }

    /// One observation, once per fresh episode, every episode carrying the
    /// same seed: a sampling model that seeds from the episode seed answers
    /// identically, one that ignores the seed does not.
    fn check_seeded_repro(&mut self, floor: Option<f64>) {
        let name = "seeded_repro";
        let range = action_range(&self.action);
        let check = match self.first_actions((0..REPRO_TRIALS).map(|_| Some(REPRO_EPISODE_SEED))) {
            Err(error) => ProbeCheck::new(name, CheckStatus::Fail, error),
            Ok(trials) => match trials_spread(&trials) {
                Some(spread) => {
                    let (status, detail) = if spread.max == 0.0 {
                        (
                            CheckStatus::Pass,
                            format!(
                                "identical actions across {REPRO_TRIALS} episodes seeded alike"
                            ),
                        )
                    } else {
                        (
                            CheckStatus::Warn,
                            format!(
                                "{REPRO_TRIALS} episodes seeded alike differ by max {:.3e}{} (mean {:.3e}, {:.1}% of {} action elements); the model does not seed from the episode seed",
                                spread.max,
                                range_fraction_note(spread.max, range),
                                spread.mean(),
                                spread.differing_fraction() * 100.0,
                                spread.elements
                            ),
                        )
                    };
                    let mut check = ProbeCheck::new(name, status, detail)
                        .metric("trials", REPRO_TRIALS as f64)
                        .metric("maxAbsDiff", spread.max)
                        .metric("meanAbsDiff", spread.mean())
                        .metric("differingFraction", spread.differing_fraction())
                        .metric("elements", spread.elements as f64);
                    if let Some(range) = range {
                        check = check
                            .metric("actionRange", range)
                            .metric("maxRangeFraction", spread.max / range);
                    }
                    if spread.max != 0.0 {
                        check = check.tag("unseeded");
                    }
                    check
                }
                None => ProbeCheck::new(
                    name,
                    CheckStatus::Fail,
                    "seeded episodes returned incomparable action shapes",
                )
                .tag("nondeterministic_shape"),
            },
        };
        self.probe.observed.seeded_reproducible = match check.status {
            CheckStatus::Pass => Some(true),
            CheckStatus::Warn => Some(false),
            _ => None,
        };
        let check = if check.status == CheckStatus::Pass {
            self.measure_seed_sensitivity(floor, check)
        } else {
            check
        };
        self.probe.record(check);
    }

    /// The other half of the seeding gate: two episodes on *different* seeds,
    /// same observation. Only a model that samples (a floor above zero) can
    /// show that the seed reaches its sampler; a deterministic one answering
    /// two seeds alike says nothing either way.
    fn measure_seed_sensitivity(&mut self, floor: Option<f64>, check: ProbeCheck) -> ProbeCheck {
        let trials = match self.first_actions(SENSITIVITY_SEEDS.map(Some)) {
            Ok(trials) => trials,
            Err(error) => {
                return check
                    .tag("seed_sensitivity_unmeasured")
                    .metric("seedSensitivityMeasured", 0.0)
                    .detail_suffix(format!(
                        "; the differing-seed leg could not run ({error}), so nothing says the seed reaches the sampler"
                    ));
            }
        };
        let Some(spread) = trials_spread(&trials) else {
            return check
                .tag("seed_sensitivity_unmeasured")
                .metric("seedSensitivityMeasured", 0.0)
                .detail_suffix(
                    "; the differing-seed leg returned incomparable action shapes".to_string(),
                );
        };
        let mut check = check
            .metric("seedSensitivityMeasured", 1.0)
            .metric("seedSpreadMaxAbsDiff", spread.max);
        match floor {
            // A sampling model whose two seeds land on the same answer is
            // dropping the seed: the same-seed Pass above measured nothing.
            Some(measured_floor) if measured_floor > 0.0 && spread.max == 0.0 => {
                self.probe.observed.seed_sensitive = Some(false);
                check.status = CheckStatus::Warn;
                check = check.tag("seed_ignored");
                check.detail_suffix(format!(
                    "; but two DIFFERENT episode seeds answered identically while the model samples (determinism floor {measured_floor:.3e}), so the seed never reaches the sampler and reproducibility here is an accident, not the seed"
                ))
            }
            Some(measured_floor) if measured_floor > 0.0 => {
                self.probe.observed.seed_sensitive = Some(true);
                check.detail_suffix(format!(
                    "; two different episode seeds answered differently (max {:.3e}), so the seed reaches the sampler",
                    spread.max
                ))
            }
            _ => check.detail_suffix(
                "; the model is deterministic, so the differing-seed leg can say nothing about whether the seed is read".to_string(),
            ),
        }
    }

    /// Input A, then B, then A again in one seeded episode: a stateless model
    /// answers the two As alike (within the floor). A drift that A,A,A at the
    /// same predict index reproduces is the per-predict seed, not state; one
    /// that a fresh episode clears is per-episode state, which a binding
    /// evicts at the episode end; one that survives it fails.
    fn check_stateless(&mut self, floor: Option<f64>) {
        let name = "stateless";
        let Some(floor) = floor else {
            self.probe.record(ProbeCheck::skipped(
                name,
                "determinism_floor did not produce a floor",
            ));
            return;
        };
        let (first, replay) =
            match self.episode(Some(EPISODE_SEED), &[OBS_SEED_A, OBS_SEED_B, OBS_SEED_A]) {
                Ok(frames) => (frames[0].clone(), frames[2].clone()),
                Err(error) => {
                    self.probe
                        .record(ProbeCheck::new(name, CheckStatus::Fail, error));
                    return;
                }
            };
        let tolerance = (floor * STATE_TOLERANCE).max(STATE_ATOL);
        let range = action_range(&self.action);
        let range_metrics = |check: ProbeCheck, delta: f64| match range {
            Some(range) => check
                .metric("actionRange", range)
                .metric("replayRangeFraction", delta / range),
            None => check,
        };
        let check = match replay_drift(frames_max_abs_diff(&first, &replay), tolerance) {
            Drift::Shape => ProbeCheck::new(
                name,
                CheckStatus::Fail,
                "A,B,A replay returned incomparable action shapes",
            )
            .tag("nondeterministic_shape"),
            // A tolerance as wide as the action range can never be exceeded,
            // so the pass would say nothing about state.
            Drift::Within(delta) if range.is_some_and(|range| tolerance >= range) => {
                let check = ProbeCheck::new(
                    name,
                    CheckStatus::Warn,
                    format!(
                        "inconclusive: replaying input A after input B drifted {delta:.3e}{}, but the tolerance {tolerance:.3e} ({STATE_TOLERANCE}x the determinism floor) covers the whole action range, so state cannot be told apart from nondeterminism",
                        range_fraction_note(delta, range)
                    ),
                )
                .tag("inconclusive")
                .metric("replayMaxAbsDiff", delta)
                .metric("tolerance", tolerance);
                range_metrics(check, delta)
            }
            Drift::Within(delta) => {
                self.probe.observed.stateless = Some(true);
                let check = ProbeCheck::new(
                    name,
                    CheckStatus::Pass,
                    format!(
                        "replaying input A after input B drifted {delta:.3e}{}, within tolerance {tolerance:.3e} ({STATE_TOLERANCE}x the determinism floor)",
                        range_fraction_note(delta, range)
                    ),
                )
                .metric("replayMaxAbsDiff", delta)
                .metric("tolerance", tolerance);
                range_metrics(check, delta)
            }
            // The replay is the episode's third predict and `first` its first,
            // so a model that seeds its sampler per predict drifts with no
            // state at all. A,A,A asks the same third predict without B: if it
            // matches the replay, B left nothing behind.
            Drift::Beyond(delta)
                if self
                    .episode(Some(EPISODE_SEED), &[OBS_SEED_A; 3])
                    .ok()
                    .and_then(|control| frames_max_abs_diff(&replay, &control[2]))
                    .is_some_and(|gap| gap <= tolerance) =>
            {
                self.probe.observed.stateless = Some(true);
                let check = ProbeCheck::new(
                    name,
                    CheckStatus::Pass,
                    format!(
                        "replaying input A after input B drifted {delta:.3e}{} from the first answer, but matched input A asked at the same predict index without B (A,A,A), so the drift is the per-predict seed, not state",
                        range_fraction_note(delta, range)
                    ),
                )
                .tag("per_predict_seed")
                .metric("replayMaxAbsDiff", delta)
                .metric("tolerance", tolerance);
                range_metrics(check, delta)
            }
            Drift::Beyond(delta) => {
                self.probe.observed.stateless = Some(false);
                let after_reset = self
                    .episode(Some(EPISODE_SEED), &[OBS_SEED_A])
                    .map(|after| frames_max_abs_diff(&first, &after[0]));
                range_metrics(stateful_check(delta, tolerance, after_reset), delta)
            }
        };
        self.probe.record(check);
    }
}
