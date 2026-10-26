//! The environment probe: handshake, describe and contract validation,
//! reproducibility, step conformance and latency, and the preview clip.
//!
//! Ported from the managed probe's env checks. The env is reached through a
//! binding with the random baseline as its model, and every step carries the
//! probe's own action, so the binding is only the 0.2 transport.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use rlmesh_spaces::{SpaceSpec as WireSpec, SpaceValue};
use serde_json::Value as Json;

use super::compare::{
    ENV_BRANCH_METADATA_KEY, ENV_STEP_SEED, IMAGE_REPLAY_ATOL, IMAGE_REPLAY_MAX_FRACTION, Replay,
    bounded_action, compare_replays, env_contract_check, first_difference, image_leaves,
    neutral_action, observation_conformance,
};
use super::preview::{PREVIEW_FRAMES, PREVIEW_MIN_FRAMES, PreviewCapture, TaggedImage};
use super::report::{CheckStatus, ProbeCheck, ProbeEnv, ProbeReport, ProbeSubject};
use super::{DEFAULT_CONNECT_TIMEOUT, Probe, peer_from_describe, percentile};
use crate::Binding;
use crate::value::space_codec;

pub(super) const ENV_CHECK_NAMES: &[&str] = &[
    "env_handshake",
    "env_spec_matches_describe",
    "env_contract",
    "env_repro",
    "env_step_conforms",
    "env_step_latency",
    "preview",
];
const ENV_RESET_SEED: i64 = 7;
/// Seeded steps replayed twice from the same reset to test env reproducibility.
const ENV_REPLAY_STEPS: usize = 10;
const ENV_REPLAY_SEED: u64 = 0x5EED_00E0;
const ENV_LATENCY_WARMUP: usize = 3;
const ENV_LATENCY_STEPS: usize = 20;

/// What [`probe_env`] needs besides the env's address.
#[derive(Debug, Clone)]
pub struct EnvProbeOptions {
    /// The image's describe envelope (its `dev.rlmesh.describe` label), which
    /// `env_spec_matches_describe` and `env_contract` hold the live env to.
    /// `None` skips both.
    pub describe: Option<Json>,
    /// Write the preview clip and still here; `None` captures none.
    pub output_dir: Option<PathBuf>,
    /// How long the env gets to answer the handshake.
    pub connect_timeout: Duration,
}

impl Default for EnvProbeOptions {
    fn default() -> Self {
        Self {
            describe: None,
            output_dir: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

/// Probe the env served at the session participant `address`. Never fails:
/// a check that cannot run is recorded as failed or skipped in the report.
pub fn probe_env(address: &str, options: &EnvProbeOptions) -> ProbeReport {
    let mut probe = Probe::new(ProbeSubject::Environment, ENV_CHECK_NAMES);
    let peer = peer_from_describe(options.describe.as_ref());
    let connect_started = Instant::now();
    let mut binding = match super::bind_env(address, options.connect_timeout) {
        Ok(binding) => binding,
        Err(error) => {
            probe.record(
                ProbeCheck::new("env_handshake", CheckStatus::Fail, error.to_string())
                    .tag("connect_failed"),
            );
            probe.skip_remaining("env_handshake failed");
            return probe.report(ProbeEnv::default(), peer);
        }
    };
    let contract = binding.contract().clone();
    let env_id = contract.id().to_string();
    probe.record(
        ProbeCheck::new(
            "env_handshake",
            CheckStatus::Pass,
            format!("env {env_id} (num_envs={})", contract.num_envs()),
        )
        .metric("connectMs", connect_started.elapsed().as_secs_f64() * 1e3)
        .metric("numEnvs", contract.num_envs() as f64),
    );
    let mut checks = EnvChecks {
        probe: &mut probe,
        binding: &mut binding,
        observation: contract.observation_space().wire().clone(),
        action: contract.action_space().wire().clone(),
        metadata: contract.to_wire().ok().and_then(|wire| wire.metadata),
        describe: options.describe.as_ref(),
    };
    checks.run(options.output_dir.as_deref());
    let _ = binding.close();
    probe.report(
        ProbeEnv {
            repo: String::new(),
            env_id,
            digest: String::new(),
        },
        peer,
    )
}

struct EnvChecks<'p, 'b> {
    probe: &'p mut Probe,
    binding: &'p mut Binding<'b>,
    observation: WireSpec,
    action: WireSpec,
    metadata: Option<rlmesh_spaces::MetaMap>,
    describe: Option<&'p Json>,
}

impl EnvChecks<'_, '_> {
    fn run(&mut self, output_dir: Option<&std::path::Path>) {
        let tags = self.metadata_json(rlmesh_adapters::v1::ENV_METADATA_KEY);
        let tags = serde_json::from_value(tags).ok();
        let tagged = super::preview::tagged_images(tags.as_ref(), &self.observation, &self.action);
        self.check_spec_matches_describe();
        self.check_contract();
        self.check_repro(&tagged);
        self.check_step_conforms();
        let mut preview = output_dir
            .and_then(|_| super::preview::select_camera(&self.observation, &tagged))
            .map(PreviewCapture::new);
        self.check_step_latency(preview.as_mut());
        match output_dir {
            Some(dir) => self.check_preview(preview, dir),
            None => self.probe.record(ProbeCheck::skipped(
                "preview",
                "no output directory to write the clip to",
            )),
        }
    }

    /// A metadata entry of the live contract as JSON; null when absent.
    fn metadata_json(&self, key: &str) -> Json {
        self.metadata
            .as_ref()
            .and_then(|metadata| metadata.get(key))
            .map(crate::lockstep::meta_json::value)
            .unwrap_or(Json::Null)
    }

    fn reset(&mut self) -> Result<SpaceValue, String> {
        let (observation, _) = self
            .binding
            .reset(Some(ENV_RESET_SEED), None)
            .map_err(|error| error.to_string())?;
        space_codec::value_to_space(&self.observation, &observation).map_err(|error| {
            format!("reset observation does not fit the observation space: {error}")
        })
    }

    /// One step under `action`: the observation, reward and whether the
    /// episode ended.
    fn step(&mut self, action: SpaceValue) -> Result<(SpaceValue, f64, bool, Duration), String> {
        let action = space_codec::value_from_space(&self.action, action)
            .map_err(|error| format!("encode action: {error}"))?;
        let started = Instant::now();
        let (observation, reward, terminated, truncated, _) = self
            .binding
            .step(Some(action))
            .map_err(|error| error.to_string())?;
        let elapsed = started.elapsed();
        let observation =
            space_codec::value_to_space(&self.observation, &observation).map_err(|error| {
                format!("step observation does not fit the observation space: {error}")
            })?;
        Ok((observation, reward, terminated || truncated, elapsed))
    }

    fn check_spec_matches_describe(&mut self) {
        let name = "env_spec_matches_describe";
        let Some(envelope) = self.describe else {
            self.probe.record(ProbeCheck::skipped(
                name,
                "the image carries no describe envelope to compare against",
            ));
            return;
        };
        let declared_spec = envelope.get("env_spec").cloned().unwrap_or(Json::Null);
        if !declared_spec.is_object() || declared_spec.get("error").is_some() {
            self.probe.record(ProbeCheck::skipped(
                name,
                "the describe envelope carries no env_spec to compare against",
            ));
            return;
        }
        let mut mismatches = Vec::new();
        for (field, live) in [
            ("observation_space", &self.observation),
            ("action_space", &self.action),
        ] {
            let declared = declared_spec.get(field).cloned().unwrap_or(Json::Null);
            let live = rlmesh_adapters::v1::space_spec_to_json(live).unwrap_or(Json::Null);
            if let Some(diff) = first_difference(&live, &declared) {
                mismatches.push(format!("{field} {diff}"));
            }
        }
        let declared_tags = envelope.get("env_tags").cloned().unwrap_or(Json::Null);
        let tags_compared = declared_tags.is_object();
        if tags_compared {
            let live_tags = self.metadata_json(rlmesh_adapters::v1::ENV_METADATA_KEY);
            if let Some(diff) = first_difference(&live_tags, &declared_tags) {
                mismatches.push(format!("env_tags {diff}"));
            }
        }
        let check = if mismatches.is_empty() {
            ProbeCheck::new(
                name,
                CheckStatus::Pass,
                if tags_compared {
                    "live observation/action spaces and env_tags match the describe envelope"
                } else {
                    "live observation/action spaces match the describe envelope (no env_tags declared)"
                },
            )
        } else {
            ProbeCheck::new(name, CheckStatus::Fail, mismatches.join("; ")).tag("env_spec_mismatch")
        };
        self.probe.record(check);
    }

    /// The contract branch this image actually serves, blocking: `join`
    /// reconciles widths, dim laws and ranges but never role *names*, so a
    /// same-width contract swap passes every other check silently.
    fn check_contract(&mut self) {
        let contracts = self
            .describe
            .and_then(|envelope| envelope.get("env_contracts"))
            .cloned()
            .unwrap_or(Json::Null);
        let served_branch = self.metadata_json(ENV_BRANCH_METADATA_KEY);
        let served_tags = self.metadata_json(rlmesh_adapters::v1::ENV_METADATA_KEY);
        self.probe
            .record(env_contract_check(&contracts, &served_branch, &served_tags));
    }

    /// Resets with a fixed seed and replays the same seeded actions twice,
    /// then compares the two trajectories leaf by leaf. State leaves (and
    /// rewards / done flags) must match exactly; image leaves get a small
    /// renderer-jitter tolerance and are ranked as a separate, lesser flag.
    fn check_repro(&mut self, tagged: &[TaggedImage]) {
        let name = "env_repro";
        let mut runs = Vec::with_capacity(2);
        for _ in 0..2 {
            match self.replay() {
                Ok(run) => runs.push(run),
                Err(error) => {
                    self.probe
                        .record(ProbeCheck::new(name, CheckStatus::Fail, error));
                    return;
                }
            }
        }
        let image_paths: Vec<Vec<String>> = if tagged.is_empty() {
            let mut paths = Vec::new();
            image_leaves(&self.observation, Vec::new(), &mut paths);
            paths
        } else {
            tagged.iter().map(|image| image.path.clone()).collect()
        };
        let Some(divergence) = compare_replays(&runs[0], &runs[1], &self.observation, &image_paths)
        else {
            self.probe.record(
                ProbeCheck::new(
                    name,
                    CheckStatus::Fail,
                    "replayed trajectories returned incomparable observation shapes",
                )
                .tag("nondeterministic_shape"),
            );
            return;
        };
        let state_ok = divergence.state_first.is_none();
        let image_ok = divergence.image_first.is_none();
        let observed = &mut self.probe.observed;
        observed.env_state_reproducible = Some(state_ok);
        observed.env_image_reproducible = Some(image_ok);
        observed.env_reset_reproducible =
            Some(divergence.state_first != Some(0) && divergence.image_first != Some(0));

        let at = |step: usize| {
            if step == 0 {
                "at reset".to_string()
            } else {
                format!("from step {step}")
            }
        };
        let state_note = match divergence.state_first {
            None => "state identical".to_string(),
            Some(step) => format!(
                "state diverges {} (max {:.3e})",
                at(step),
                divergence.state_max
            ),
        };
        let image_note = if image_paths.is_empty() {
            "no image leaves".to_string()
        } else {
            match divergence.image_first {
                None => format!(
                    "images within renderer tolerance ({:.2}% of pixels beyond ±{IMAGE_REPLAY_ATOL:.0}, max {:.3e})",
                    divergence.image_over_fraction * 100.0,
                    divergence.image_max
                ),
                Some(step) => format!(
                    "images diverge {} ({:.2}% of pixels beyond ±{IMAGE_REPLAY_ATOL:.0} > {:.0}% renderer tolerance, max {:.3e})",
                    at(step),
                    divergence.image_over_fraction * 100.0,
                    IMAGE_REPLAY_MAX_FRACTION * 100.0,
                    divergence.image_max
                ),
            }
        };
        let detail = format!(
            "reset(seed={ENV_RESET_SEED}) + {} seeded steps replayed twice: {state_note}; {image_note}",
            divergence.steps
        );
        let mut check = ProbeCheck::new(
            name,
            if state_ok && image_ok {
                CheckStatus::Pass
            } else {
                CheckStatus::Warn
            },
            detail,
        )
        .metric("steps", divergence.steps as f64)
        .metric("stateMaxAbsDiff", divergence.state_max)
        .metric("imageMaxAbsDiff", divergence.image_max)
        .metric("imageOverToleranceFraction", divergence.image_over_fraction)
        .metric("imageTolerance", IMAGE_REPLAY_ATOL)
        .metric("imageMaxFraction", IMAGE_REPLAY_MAX_FRACTION);
        if let Some(step) = divergence.state_first {
            check = check
                .tag("nonreproducible_state")
                .metric("stateDivergesAt", step as f64);
        }
        if let Some(step) = divergence.image_first {
            check = check
                .tag("nonreproducible_image")
                .metric("imageDivergesAt", step as f64);
        }
        let check = check.metric(
            "resetMs",
            runs.iter().map(|run| run.reset_ms).fold(0.0, f64::max),
        );
        self.probe.record(check);
    }

    /// One reset plus up to `ENV_REPLAY_STEPS` seeded steps, stopping at the
    /// first episode end so both runs see the same schedule.
    fn replay(&mut self) -> Result<Replay, String> {
        let reset_started = Instant::now();
        let first = self.reset()?;
        let mut replay = Replay {
            observations: vec![first],
            rewards: Vec::new(),
            dones: Vec::new(),
            reset_ms: reset_started.elapsed().as_secs_f64() * 1e3,
        };
        for step in 0..ENV_REPLAY_STEPS {
            let seed = ENV_REPLAY_SEED + (step as u64) * 1_000;
            let action = rlmesh_spaces::sample_seeded(&self.action, seed)
                .map_err(|err| format!("sample action: {err}"))?;
            let (observation, reward, done, _) = self.step(action)?;
            replay.observations.push(observation);
            replay.rewards.push(reward);
            replay.dones.push(done);
            if done {
                break;
            }
        }
        Ok(replay)
    }

    fn check_step_conforms(&mut self) {
        let name = "env_step_conforms";
        let stepped = match neutral_action(&self.action) {
            Ok(action) => match self.reset() {
                Ok(_) => self.step(action),
                Err(error) => Err(format!("reset before step: {error}")),
            },
            Err(error) => Err(error),
        };
        let check = match stepped {
            Ok((observation, _, _, _)) => {
                observation_conformance(name, &[observation], &self.observation)
                    .metric("lanes", 1.0)
            }
            Err(error) => ProbeCheck::new(name, CheckStatus::Fail, error),
        };
        self.probe.record(check);
    }

    /// Times `ENV_LATENCY_STEPS` random-action steps; with a preview capture
    /// attached, keeps stepping (untimed) until it has its frames.
    fn check_step_latency(&mut self, mut preview: Option<&mut PreviewCapture>) {
        let name = "env_step_latency";
        match self.reset() {
            // The reset frame leads the preview: the undisturbed scene is the still.
            Ok(observation) => {
                if let Some(capture) = preview.as_deref_mut() {
                    capture.capture(&observation, &self.observation);
                }
            }
            Err(error) => {
                self.probe.record(ProbeCheck::new(
                    name,
                    CheckStatus::Fail,
                    format!("reset before stepping: {error}"),
                ));
                return;
            }
        }
        let mut wall_ms = Vec::with_capacity(ENV_LATENCY_STEPS);
        let mut resets = 0u32;
        let mut failure = None;
        let timed_steps =
            (ENV_LATENCY_WARMUP as u64)..(ENV_LATENCY_WARMUP + ENV_LATENCY_STEPS) as u64;
        let max_steps = ENV_LATENCY_WARMUP
            + if preview.is_some() {
                ENV_LATENCY_STEPS.max(PREVIEW_FRAMES)
            } else {
                ENV_LATENCY_STEPS
            };
        for step in 0..max_steps as u64 {
            if step >= timed_steps.end && preview.as_ref().is_none_or(|capture| capture.done()) {
                break;
            }
            // A fresh draw per step: the preview rides this loop, and a scene
            // that never moves is not worth looking at.
            let action = match bounded_action(&self.action, ENV_STEP_SEED + step) {
                Ok(action) => action,
                Err(error) => {
                    failure = Some(format!("step {}: {error}", step + 1));
                    break;
                }
            };
            match self.step(action) {
                Ok((observation, _, done, elapsed)) => {
                    if timed_steps.contains(&step) {
                        wall_ms.push(elapsed.as_secs_f64() * 1e3);
                    }
                    if let Some(capture) = preview.as_deref_mut() {
                        capture.capture(&observation, &self.observation);
                    }
                    if done {
                        resets += 1;
                        if let Err(error) = self.reset() {
                            failure = Some(format!("reset after episode end: {error}"));
                            break;
                        }
                    }
                }
                Err(error) => {
                    failure = Some(format!("step {}: {error}", step + 1));
                    break;
                }
            }
        }
        let check = match failure {
            Some(error) if wall_ms.is_empty() => ProbeCheck::new(name, CheckStatus::Fail, error),
            outcome => {
                wall_ms.sort_by(f64::total_cmp);
                let p50 = percentile(&wall_ms, 0.5);
                let p95 = percentile(&wall_ms, 0.95);
                let (status, detail) = match outcome {
                    Some(error) => (
                        CheckStatus::Warn,
                        format!(
                            "{} of {ENV_LATENCY_STEPS} steps timed (p50 {p50:.1} ms, p95 {p95:.1} ms) before {error}",
                            wall_ms.len()
                        ),
                    ),
                    None => (
                        CheckStatus::Pass,
                        format!(
                            "{ENV_LATENCY_STEPS} steps after {ENV_LATENCY_WARMUP} warmup: p50 {p50:.1} ms, p95 {p95:.1} ms"
                        ),
                    ),
                };
                ProbeCheck::new(name, status, detail)
                    .metric("p50Ms", p50)
                    .metric("p95Ms", p95)
                    .metric("steps", wall_ms.len() as f64)
                    .metric("resets", f64::from(resets))
            }
        };
        self.probe.record(check);
    }

    fn check_preview(&mut self, preview: Option<PreviewCapture>, output_dir: &std::path::Path) {
        let name = "preview";
        let Some(preview) = preview else {
            self.probe
                .record(ProbeCheck::skipped(name, "no image observation"));
            return;
        };
        let camera = preview.camera_name().to_string();
        let frames = preview.frames();
        let check = match preview.write(output_dir) {
            Ok(Some(written)) => {
                let (status, detail) = if frames < PREVIEW_MIN_FRAMES {
                    (
                        CheckStatus::Warn,
                        format!(
                            "only {frames} of {PREVIEW_FRAMES} frames captured from {camera} ({}x{})",
                            written.width, written.height
                        ),
                    )
                } else {
                    (
                        CheckStatus::Pass,
                        format!(
                            "{frames} frames from {camera} at {}x{}, {} fps",
                            written.width, written.height, written.fps
                        ),
                    )
                };
                let check = ProbeCheck::new(name, status, detail)
                    .metric("frames", f64::from(written.frames))
                    .metric("width", f64::from(written.width))
                    .metric("height", f64::from(written.height));
                self.probe.preview = Some(written);
                check
            }
            Ok(None) => ProbeCheck::new(
                name,
                CheckStatus::Warn,
                format!("no frames captured from {camera}"),
            )
            .metric("frames", 0.0),
            Err(error) => ProbeCheck::new(name, CheckStatus::Fail, error)
                .tag("encode_failed")
                .metric("frames", frames as f64),
        };
        self.probe.record(check);
    }
}
