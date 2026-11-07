use crate::data::{EnvError, EnvErrorCode};
use rlmesh_proto::Edition;

use super::lanes::{LaneEnv, LaneReply, batch_infos, fold_phases};
use super::legacy::{Env, VectorEnv};
use super::types::{
    CloseRequest, RenderRequest, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
};
use crate::binding::spaces::{self, RenderResult};
use rlmesh_proto::{EndpointPhases, elapsed_ns};
use rlmesh_spaces::spaces::{PolicyOutcome, ValidationPolicy};
use std::collections::{BTreeMap, HashSet};
use std::time::Instant;

/// One value-conformance warning surfaced in the info map.
struct ConformanceWarning {
    kind: String,
    path: String,
    detail: String,
}

/// Resolve the serving-side validation policy from `RLMESH_VALIDATION_POLICY`
/// (`strict`/`off`; default and any other value are `warn`).
fn validation_policy_from_env() -> ValidationPolicy {
    match std::env::var("RLMESH_VALIDATION_POLICY") {
        Ok(raw) => match raw.trim().to_ascii_lowercase().as_str() {
            "strict" => ValidationPolicy::Strict,
            "off" => ValidationPolicy::Off,
            _ => ValidationPolicy::Warn,
        },
        Err(_) => ValidationPolicy::Warn,
    }
}

/// Merge conformance warnings into an info map under the session edition's
/// reserved key.
fn inject_conformance_warnings(
    edition: Edition,
    info: &mut Option<spaces::MetaMap>,
    warnings: Vec<ConformanceWarning>,
) {
    if warnings.is_empty() {
        return;
    }
    let entries = warnings
        .into_iter()
        .map(|warning| {
            spaces::MetaValue::Map(BTreeMap::from([
                ("kind".to_string(), spaces::MetaValue::String(warning.kind)),
                ("path".to_string(), spaces::MetaValue::String(warning.path)),
                (
                    "detail".to_string(),
                    spaces::MetaValue::String(warning.detail),
                ),
            ]))
        })
        .collect();
    info.get_or_insert_with(BTreeMap::new).insert(
        rlmesh_proto::defaults(edition)
            .conformance_warning_info_key
            .to_string(),
        spaces::MetaValue::List(entries),
    );
}

/// Serving-side value conformance: validate observations/actions against their
/// declared space under the active policy. Structural deviations always
/// reject; range deviations follow the policy (default `warn`, recorded once
/// per `(kind, path)`). Shared by both adapters; lock-free on the accept path.
struct Conformance {
    policy: ValidationPolicy,
    /// `(lane, kind, path)` already reported for the server's life: once per
    /// lane of a lane endpoint, once for all lanes of a vector env (lane 0).
    /// Behind a lock because lane ops run concurrently through `&self`.
    warned: std::sync::Mutex<HashSet<(usize, String, String)>>,
    /// The edition the current session runs at (see
    /// the session's edition pin); the warning key is its promise.
    edition: std::sync::Mutex<Edition>,
}

impl Conformance {
    fn from_env() -> Self {
        Self {
            policy: validation_policy_from_env(),
            warned: std::sync::Mutex::new(HashSet::new()),
            edition: std::sync::Mutex::new(Edition::current()),
        }
    }

    fn edition(&self) -> Edition {
        *self
            .edition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn enforce(
        &self,
        space: &spaces::SpaceSpec,
        value: &spaces::SpaceValue,
        kind: &str,
        lane: usize,
        warnings: &mut Vec<ConformanceWarning>,
    ) -> Result<(), EnvError> {
        match self.policy.check(space, value) {
            PolicyOutcome::Accept => Ok(()),
            PolicyOutcome::Reject(err) => {
                let code = if kind == "action" {
                    EnvErrorCode::InvalidAction
                } else {
                    EnvErrorCode::Internal
                };
                Err(EnvError::new(code, err.to_string()))
            }
            PolicyOutcome::Warn(err) => {
                let path = err.path().to_string();
                let first_time = self
                    .warned
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .insert((lane, kind.to_string(), path.clone()));
                if first_time {
                    warnings.push(ConformanceWarning {
                        kind: kind.to_string(),
                        path,
                        detail: err.to_string(),
                    });
                }
                Ok(())
            }
        }
    }

    fn enforce_all(
        &self,
        space: &spaces::SpaceSpec,
        values: &[spaces::SpaceValue],
        kind: &str,
        lanes: &[usize],
        warnings: &mut Vec<ConformanceWarning>,
    ) -> Result<(), EnvError> {
        for (pos, value) in values.iter().enumerate() {
            let lane = lanes.get(pos).copied().unwrap_or(0);
            self.enforce(space, value, kind, lane, warnings)?;
        }
        Ok(())
    }
}

/// Check a step's per-lane results (every per-lane vector `width` long) and
/// its observations, folding every conformance warning into its info.
fn finish_step(
    conformance: &Conformance,
    observation_space: &spaces::SpaceSpec,
    mut result: super::types::StepResult,
    width: usize,
    lanes: &[usize],
    mut warnings: Vec<ConformanceWarning>,
) -> Result<super::types::StepResult, EnvError> {
    validate_count(&result.observations, width, "observations")?;
    validate_count(&result.terminated, width, "terminated values")?;
    validate_count(&result.truncated, width, "truncated values")?;
    validate_count(&result.rewards, width, "rewards values")?;
    conformance.enforce_all(
        observation_space,
        &result.observations,
        "observation",
        lanes,
        &mut warnings,
    )?;
    inject_conformance_warnings(conformance.edition(), &mut result.info, warnings);
    Ok(result)
}

/// Check a reset's per-lane observations, folding conformance warnings into
/// its info.
fn finish_reset(
    conformance: &Conformance,
    observation_space: &spaces::SpaceSpec,
    mut result: VectorResetResult,
    width: usize,
    lanes: &[usize],
) -> Result<VectorResetResult, EnvError> {
    validate_count(&result.observations, width, "observations")?;
    let mut warnings = Vec::new();
    conformance.enforce_all(
        observation_space,
        &result.observations,
        "observation",
        lanes,
        &mut warnings,
    )?;
    inject_conformance_warnings(conformance.edition(), &mut result.info, warnings);
    Ok(result)
}

/// Check `width` per-lane actions: their count and their conformance.
fn check_actions(
    conformance: &Conformance,
    action_space: &spaces::SpaceSpec,
    actions: &[spaces::SpaceValue],
    width: usize,
    lanes: &[usize],
    warnings: &mut Vec<ConformanceWarning>,
) -> Result<(), EnvError> {
    validate_action_count(actions, width)?;
    conformance.enforce_all(action_space, actions, "action", lanes, warnings)
}

/// What a native op measured: its queue wait, the env call and the env's
/// own split of it.
struct Call {
    queue_ns: u64,
    call_ns: u64,
    inner: EndpointPhases,
}

impl Call {
    /// The wire split, with this layer's decode and encode around the call.
    fn phases(self, decode_ns: u64, encode_ns: u64) -> EndpointPhases {
        let mut phases = EndpointPhases::nest(decode_ns, self.call_ns, encode_ns, self.inner);
        phases.queue_ns = self.queue_ns;
        phases
    }
}

/// Bridges a lockstep [`VectorEnv`] (a natively batched env such as a gym
/// vector env) to the transport trait. Ops serialize on the env; a step that
/// names lanes is rejected (`supports_lanes` is false).
#[doc(hidden)]
pub struct WireEnvAdapter<E> {
    inner: tokio::sync::Mutex<E>,
    observation_space: spaces::SpaceSpec,
    action_space: spaces::SpaceSpec,
    env_contract: spaces::EnvContract,
    num_envs: usize,
    conformance: Conformance,
}

impl<E: VectorEnv> WireEnvAdapter<E> {
    /// Wrap a [`VectorEnv`] for the wire layer.
    #[doc(hidden)]
    pub fn new(inner: E) -> Self {
        Self {
            observation_space: inner.observation_space().clone(),
            action_space: inner.action_space().clone(),
            env_contract: inner.env_contract().clone(),
            num_envs: inner.num_envs(),
            inner: tokio::sync::Mutex::new(inner),
            conformance: Conformance::from_env(),
        }
    }

    /// Reset `lanes` (empty: the whole vector), checked and conformed.
    async fn reset_native(
        &self,
        lanes: &[u32],
        seeds: Vec<i64>,
        options: Option<spaces::MetaMap>,
        timeout_ms: i64,
    ) -> Result<(VectorResetResult, EndpointPhases), EnvError> {
        let request = VectorResetRequest {
            seeds,
            options,
            timeout_ms,
            env_indices: lane_indices_to_native(lanes),
        };
        // A reset naming lanes replies only those lanes; a whole-vector reset
        // replies full width.
        let width = if lanes.is_empty() {
            self.num_envs
        } else {
            lanes.len()
        };
        let lock_started = Instant::now();
        let mut env = self.inner.lock().await;
        let queue_ns = elapsed_ns(lock_started);
        let call_started = Instant::now();
        // An env that cannot reset individual sub-envs inherits the rejecting
        // `reset_subset` default, so a partial reset fails loud here rather
        // than silently resetting the whole vector.
        let result = if lanes.is_empty() {
            env.reset(request).await
        } else {
            env.reset_subset(request).await
        }
        .map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let inner = env.take_last_phases();
        drop(env);
        let result = finish_reset(
            &self.conformance,
            &self.observation_space,
            result,
            width,
            &[],
        )?;
        Ok((
            result,
            Call {
                queue_ns,
                call_ns,
                inner,
            }
            .phases(0, 0),
        ))
    }

    /// Step the whole vector, checked and conformed.
    async fn step_native(
        &self,
        lanes: &[u32],
        actions: Vec<spaces::SpaceValue>,
        timeout_ms: i64,
    ) -> Result<(super::types::StepResult, EndpointPhases), EnvError> {
        if !lanes.is_empty() {
            return Err(EnvError::new(
                EnvErrorCode::Unsupported,
                "this environment steps its whole vector in lockstep; a step naming lanes \
                 (StepRequest.env_indices) needs a lane endpoint",
            ));
        }
        let mut warnings = Vec::new();
        check_actions(
            &self.conformance,
            &self.action_space,
            &actions,
            self.num_envs,
            &[],
            &mut warnings,
        )?;
        let lock_started = Instant::now();
        let mut env = self.inner.lock().await;
        let queue_ns = elapsed_ns(lock_started);
        let call_started = Instant::now();
        let result = env
            .step(VectorStepRequest {
                actions,
                timeout_ms,
            })
            .await
            .map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let inner = env.take_last_phases();
        drop(env);
        let result = finish_step(
            &self.conformance,
            &self.observation_space,
            result,
            self.num_envs,
            &[],
            warnings,
        )?;
        Ok((
            result,
            Call {
                queue_ns,
                call_ns,
                inner,
            }
            .phases(0, 0),
        ))
    }

    /// Render lane `lane` (`None`: the env's choice).
    async fn render_native(
        &self,
        lane: Option<usize>,
        timeout_ms: i64,
    ) -> Result<(RenderResult, EndpointPhases), EnvError> {
        let lock_started = Instant::now();
        let mut env = self.inner.lock().await;
        let queue_ns = elapsed_ns(lock_started);
        let call_started = Instant::now();
        let result = env
            .render(RenderRequest {
                env_index: lane,
                timeout_ms,
            })
            .await
            .map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let inner = env.take_last_phases();
        Ok((
            result,
            Call {
                queue_ns,
                call_ns,
                inner,
            }
            .phases(0, 0),
        ))
    }

    async fn close_native(&self) -> Result<super::types::CloseResult, EnvError> {
        self.inner
            .lock()
            .await
            .close(CloseRequest {
                wait_for_episodes: false,
            })
            .await
            .map_err(gym_error_to_env_error)
    }
}

/// Bridges a [`LaneEnv`] (N scalar envs, one actor each) to the transport
/// trait: a request naming lanes fans out to those actors and gathers, a
/// whole-vector request covers every lane. Lane ops run concurrently.
#[doc(hidden)]
pub struct WireLaneAdapter {
    lanes: LaneEnv,
    conformance: Conformance,
}

impl WireLaneAdapter {
    /// Serve `envs` as the lanes of one endpoint. Errors if the lanes disagree
    /// on their env contract.
    pub fn new<E: Env + 'static>(envs: Vec<E>) -> Result<Self, spaces::EnvRuntimeError> {
        Ok(Self {
            lanes: LaneEnv::new(envs)?,
            conformance: Conformance::from_env(),
        })
    }

    pub(crate) fn track_abandoned(&mut self, threads: super::lanes::JoinThreads) {
        self.lanes.track_abandoned(threads);
    }

    /// The lanes a request covers: the ones it names, else every lane.
    fn lanes_for(&self, env_indices: &[u32]) -> Vec<usize> {
        if env_indices.is_empty() {
            (0..self.lanes.num_envs()).collect()
        } else {
            env_indices.iter().map(|&lane| lane as usize).collect()
        }
    }

    /// Reset `lanes` (empty: every lane) side by side, checked and conformed.
    async fn reset_native(
        &self,
        lanes: &[u32],
        seeds: Vec<i64>,
        options: Option<spaces::MetaMap>,
        timeout_ms: i64,
    ) -> Result<(VectorResetResult, EndpointPhases), EnvError> {
        let lanes = self.lanes_for(lanes);
        // Each scalar lane gets its own slice of a per-lane option list, so a
        // list that does not cover the lanes fails the whole reset here rather
        // than per lane.
        let lane_options = (0..lanes.len())
            .map(|pos| lane_reset_options(options.as_ref(), pos, lanes.len()))
            .collect::<Result<Vec<_>, EnvError>>()?;

        let call_started = Instant::now();
        let results = futures::future::join_all(lanes.iter().enumerate().map(|(pos, &lane)| {
            let request = spaces::request::ResetRequest {
                seed: seeds.get(pos).copied(),
                options: lane_options[pos].clone(),
                timeout_ms,
            };
            self.lanes.reset(lane, request)
        }))
        .await;
        let call_ns = elapsed_ns(call_started);
        let mut observations = Vec::with_capacity(lanes.len());
        let mut infos = Vec::with_capacity(lanes.len());
        let mut phases = Vec::with_capacity(lanes.len());
        for (result, lane_phases) in gather_lane_results(results, &lanes, "reset")? {
            observations.extend(result.observation);
            infos.push(result.info);
            phases.push(lane_phases);
        }
        let result = finish_reset(
            &self.conformance,
            self.lanes.observation_space(),
            VectorResetResult {
                observations,
                info: batch_infos(infos, None),
                episode_ids: Vec::new(),
            },
            lanes.len(),
            &lanes,
        )?;
        let inner = fold_phases(phases);
        Ok((
            result,
            Call {
                queue_ns: inner.queue_ns,
                call_ns,
                inner,
            }
            .phases(0, 0),
        ))
    }

    /// Step `lanes` (empty: every lane) side by side, checked and conformed.
    async fn step_native(
        &self,
        lanes: &[u32],
        actions: Vec<spaces::SpaceValue>,
        timeout_ms: i64,
    ) -> Result<(super::types::StepResult, EndpointPhases), EnvError> {
        let lanes = self.lanes_for(lanes);
        let mut warnings = Vec::new();
        check_actions(
            &self.conformance,
            self.lanes.action_space(),
            &actions,
            lanes.len(),
            &lanes,
            &mut warnings,
        )?;
        let call_started = Instant::now();
        let results =
            futures::future::join_all(lanes.iter().zip(actions).map(|(&lane, action)| {
                self.lanes.step(
                    lane,
                    spaces::request::StepRequest {
                        action: Some(action),
                        timeout_ms,
                    },
                )
            }))
            .await;
        let call_ns = elapsed_ns(call_started);
        let mut result = super::types::StepResult::default();
        let mut infos = Vec::with_capacity(lanes.len());
        let mut done = Vec::with_capacity(lanes.len());
        let mut phases = Vec::with_capacity(lanes.len());
        for (step, lane_phases) in gather_lane_results(results, &lanes, "step")? {
            result.observations.extend(step.observation);
            result.rewards.push(step.reward);
            result.terminated.push(step.terminated);
            result.truncated.push(step.truncated);
            done.push(step.terminated || step.truncated);
            infos.push(step.info);
            phases.push(lane_phases);
        }
        result.info = batch_infos(infos, Some(&done));
        let result = finish_step(
            &self.conformance,
            self.lanes.observation_space(),
            result,
            lanes.len(),
            &lanes,
            warnings,
        )?;
        let inner = fold_phases(phases);
        Ok((
            result,
            Call {
                queue_ns: inner.queue_ns,
                call_ns,
                inner,
            }
            .phases(0, 0),
        ))
    }

    /// Render lane `lane`.
    async fn render_native(
        &self,
        lane: usize,
        timeout_ms: i64,
    ) -> Result<(RenderResult, EndpointPhases), EnvError> {
        let request = RenderRequest {
            env_index: Some(lane),
            timeout_ms,
        };
        let call_started = Instant::now();
        let (result, inner) = self
            .lanes
            .render(lane, request)
            .await
            .map_err(gym_error_to_env_error)?;
        Ok((
            result,
            Call {
                queue_ns: inner.queue_ns,
                call_ns: elapsed_ns(call_started),
                inner,
            }
            .phases(0, 0),
        ))
    }

    async fn close_native(&self) -> Result<(), EnvError> {
        self.lanes
            .close(CloseRequest {
                wait_for_episodes: false,
            })
            .await
            .map_err(gym_error_to_env_error)
    }
}

/// The lockstep path serves both adapters natively (§12.4): the same checks,
/// conformance and env calls as their wire shells, without the v1 messages.
mod served {
    use async_trait::async_trait;

    use super::{EnvError, VectorEnv, WireEnvAdapter, WireLaneAdapter};
    use crate::EnvironmentError;
    use crate::binding::spaces;
    use crate::lockstep::{Reset, Served, Step};

    fn failure(error: EnvError) -> EnvironmentError {
        let class = error.debug_info.clone();
        EnvironmentError {
            class,
            ..EnvironmentError::from_wire(error)
        }
    }

    fn reset(result: super::VectorResetResult) -> Reset {
        Reset {
            observations: result.observations,
            info: result.info,
        }
    }

    fn step(result: super::super::types::StepResult) -> Step {
        Step {
            observations: result.observations,
            rewards: result.rewards,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        }
    }

    fn frame(result: super::RenderResult) -> Option<Vec<u8>> {
        result.frame.map(|frame| frame.frame)
    }

    fn contract(contract: &spaces::EnvContract, num_envs: usize) -> spaces::EnvContract {
        spaces::EnvContract {
            num_envs: u32::try_from(num_envs).unwrap_or(u32::MAX),
            ..contract.clone()
        }
    }

    #[async_trait]
    impl<E: VectorEnv + 'static> Served for WireEnvAdapter<E> {
        fn contract(&self) -> spaces::EnvContract {
            contract(&self.env_contract, self.num_envs)
        }

        fn subset_step(&self) -> bool {
            false
        }

        async fn reset(
            &self,
            lanes: &[u32],
            seeds: Vec<i64>,
            options: Option<spaces::MetaMap>,
        ) -> Result<Reset, EnvironmentError> {
            let (result, _) = self
                .reset_native(lanes, seeds, options, 0)
                .await
                .map_err(failure)?;
            Ok(reset(result))
        }

        async fn step(
            &self,
            lanes: &[u32],
            actions: Vec<spaces::SpaceValue>,
        ) -> Result<Step, EnvironmentError> {
            let (result, _) = self.step_native(lanes, actions, 0).await.map_err(failure)?;
            Ok(step(result))
        }

        async fn render(&self, lane: u32) -> Result<Option<Vec<u8>>, EnvironmentError> {
            let (result, _) = self
                .render_native(Some(lane as usize), 0)
                .await
                .map_err(failure)?;
            Ok(frame(result))
        }

        async fn close(&self) -> Result<(), EnvironmentError> {
            self.close_native().await.map(|_| ()).map_err(failure)
        }
    }

    #[async_trait]
    impl Served for WireLaneAdapter {
        fn contract(&self) -> spaces::EnvContract {
            contract(self.lanes.env_contract(), self.lanes.num_envs())
        }

        fn subset_step(&self) -> bool {
            true
        }

        async fn reset(
            &self,
            lanes: &[u32],
            seeds: Vec<i64>,
            options: Option<spaces::MetaMap>,
        ) -> Result<Reset, EnvironmentError> {
            let (result, _) = self
                .reset_native(lanes, seeds, options, 0)
                .await
                .map_err(failure)?;
            Ok(reset(result))
        }

        async fn step(
            &self,
            lanes: &[u32],
            actions: Vec<spaces::SpaceValue>,
        ) -> Result<Step, EnvironmentError> {
            let (result, _) = self.step_native(lanes, actions, 0).await.map_err(failure)?;
            Ok(step(result))
        }

        async fn render(&self, lane: u32) -> Result<Option<Vec<u8>>, EnvironmentError> {
            let (result, _) = self
                .render_native(lane as usize, 0)
                .await
                .map_err(failure)?;
            Ok(frame(result))
        }

        async fn close(&self) -> Result<(), EnvironmentError> {
            self.close_native().await.map_err(failure)
        }
    }
}

/// Fold a lane batch's per-lane results into the request's one outcome.
///
/// Every lane ran, so no lane's error may hide another's, and a request reported
/// as recoverable must be safe to retry as is:
/// - any lane failing fatally ends the session (that lane's error);
/// - lanes that all failed recoverably fail only the request, since no lane
///   moved;
/// - a recoverable failure next to lanes that succeeded ends the session: those
///   lanes moved on but their transitions never reach the client or the episode
///   tracker, so retrying would replay them and stepping on would desync the
///   accounting. The client's next reset opens a fresh session.
fn gather_lane_results<T>(
    results: Vec<LaneReply<T>>,
    lanes: &[usize],
    op: &str,
) -> Result<Vec<(T, EndpointPhases)>, EnvError> {
    if results.iter().all(Result::is_ok) {
        return Ok(results.into_iter().filter_map(Result::ok).collect());
    }
    let mut failed = Vec::new();
    let mut moved = Vec::new();
    for (result, &lane) in results.into_iter().zip(lanes) {
        match result {
            Ok(_) => moved.push(lane),
            Err(error) => failed.push((lane, gym_error_to_env_error(error))),
        }
    }
    if let Some(index) = failed.iter().position(|(_, error)| !error.is_recoverable) {
        return Err(failed.swap_remove(index).1);
    }
    let (lane, first) = failed.swap_remove(0);
    if moved.is_empty() {
        return Err(first);
    }
    Err(EnvError::new(
        EnvErrorCode::Internal,
        format!(
            "lane {lane} {op} failed with a recoverable error ({}) while lanes {moved:?} \
             completed theirs; a partly applied lane batch cannot be retried, so the session \
             ends",
            first.message
        ),
    ))
}

fn gym_error_to_env_error(error: spaces::EnvRuntimeError) -> EnvError {
    match error {
        spaces::EnvRuntimeError::InvalidSpace(message)
        | spaces::EnvRuntimeError::InvalidValue(message) => {
            EnvError::new(EnvErrorCode::InvalidAction, message)
        }
        spaces::EnvRuntimeError::Runtime(message) => EnvError::new(EnvErrorCode::Internal, message),
        // Still an internal failure, but the env vouched that it can keep
        // serving: the flag (not the code) keeps the session open.
        spaces::EnvRuntimeError::Recoverable(message) => EnvError {
            is_recoverable: true,
            ..EnvError::new(EnvErrorCode::Internal, message)
        },
        // The builtin class rides in `debug_info` to the lockstep reply.
        spaces::EnvRuntimeError::Raised {
            class,
            message,
            recoverable,
        } => EnvError {
            is_recoverable: recoverable,
            debug_info: Some(class),
            ..EnvError::new(EnvErrorCode::Internal, message)
        },
        // EnvRuntimeError is #[non_exhaustive]; treat unknown variants as internal.
        other => EnvError::new(EnvErrorCode::Internal, other.to_string()),
    }
}

/// Convert wire lane indices (uint32) to the native i32 representation. Lane
/// offsets always fit i32; an unrepresentable value is clamped rather than
/// wrapped so a foreign index is rejected loudly downstream.
fn lane_indices_to_native(env_indices: &[u32]) -> Vec<i32> {
    env_indices
        .iter()
        .map(|&index| i32::try_from(index).unwrap_or(i32::MAX))
        .collect()
}

pub(super) fn validate_action_count(
    actions: &[spaces::SpaceValue],
    num_envs: usize,
) -> std::result::Result<(), EnvError> {
    if actions.len() == num_envs {
        return Ok(());
    }
    Err(EnvError::new(
        EnvErrorCode::InvalidAction,
        format!("expected {num_envs} actions, got {}", actions.len()),
    ))
}

pub(super) fn validate_count<T>(
    values: &[T],
    num_envs: usize,
    label: &str,
) -> std::result::Result<(), EnvError> {
    if values.len() == num_envs {
        return Ok(());
    }
    Err(EnvError::new(
        EnvErrorCode::Internal,
        format!("env returned {} {label} for {num_envs} envs", values.len()),
    ))
}

/// The reset options lane `pos` of a `lanes`-wide fan-out receives.
///
/// The runtime sends the reserved `trial_index` option as a list in lane order
/// when a reset restarts several lanes (see
/// [`crate::run::runtime::reset_options_for`]), but each lane here is a scalar env
/// that gets its own `reset`: handing it the whole list leaves
/// `trial_index(options, None)` unable to read an ordinal. Narrow the list to
/// this lane's element; every other key rides through unchanged. A list that
/// does not cover the lanes it was sent for is a malformed request, not a lane
/// to silently leave unsequenced.
fn lane_reset_options(
    options: Option<&spaces::MetaMap>,
    pos: usize,
    lanes: usize,
) -> Result<Option<spaces::MetaMap>, EnvError> {
    let Some(options) = options else {
        return Ok(None);
    };
    let Some(spaces::MetaValue::List(trials)) =
        options.get(crate::run::runtime::TRIAL_INDEX_OPTION)
    else {
        return Ok(Some(options.clone()));
    };
    let Some(trial) = trials.get(pos) else {
        return Err(EnvError::new(
            EnvErrorCode::InvalidAction,
            format!(
                "reset option `{}` is a list of {} entries but this reset covers {lanes} lanes; \
                 send one ordinal per lane, in lane order",
                crate::run::runtime::TRIAL_INDEX_OPTION,
                trials.len(),
            ),
        ));
    };
    let mut sliced = options.clone();
    sliced.insert(
        crate::run::runtime::TRIAL_INDEX_OPTION.to_string(),
        trial.clone(),
    );
    Ok(Some(sliced))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_actions_are_recoverable_before_native_callbacks() {
        use spaces::spaces::{DictSpaceBuilder, DiscreteBuilder};
        let child = DictSpaceBuilder::new()
            .insert("choice", DiscreteBuilder::new(3).build().unwrap())
            .build()
            .unwrap();
        let space = DictSpaceBuilder::new()
            .insert("nested", child)
            .build()
            .unwrap();
        for policy in [
            ValidationPolicy::Warn,
            ValidationPolicy::Strict,
            ValidationPolicy::Off,
        ] {
            let conformance = Conformance {
                policy,
                ..Conformance::from_env()
            };
            for (fields, text) in [
                (BTreeMap::new(), "$.nested: missing key 'choice'"),
                (
                    BTreeMap::from([
                        ("choice".into(), spaces::SpaceValue::Discrete(1)),
                        ("extra".into(), spaces::SpaceValue::Discrete(0)),
                    ]),
                    "$.nested: unexpected key 'extra'",
                ),
            ] {
                let action = spaces::SpaceValue::Dict(BTreeMap::from([(
                    "nested".into(),
                    spaces::SpaceValue::Dict(fields),
                )]));
                // Both lane and vector adapters use this check before dispatch.
                let error = conformance
                    .enforce_all(&space, &[action], "action", &[0], &mut Vec::new())
                    .unwrap_err();
                assert_eq!(error.code, EnvErrorCode::InvalidAction);
                assert!(error.is_recoverable);
                assert_eq!(error.message, text);
            }
            let action = spaces::SpaceValue::Dict(BTreeMap::from([(
                "nested".into(),
                spaces::SpaceValue::Dict(BTreeMap::from([(
                    "choice".into(),
                    spaces::SpaceValue::Discrete(1),
                )])),
            )]));
            assert!(
                conformance
                    .enforce_all(&space, &[action], "action", &[0], &mut Vec::new())
                    .is_ok()
            );
        }
    }
}
