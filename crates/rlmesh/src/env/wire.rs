use async_trait::async_trait;
use rlmesh_grpc::env::{
    CloseEnvsResponse as ProtoCloseResponse, Environment, RenderRequest as ProtoRenderRequest,
    RenderResponse as ProtoRenderResponse, ResetRequest as ProtoResetRequest,
    ResetResponse as ProtoResetResponse, StepRequest as ProtoStepRequest,
    StepResponse as ProtoStepResponse,
};
use rlmesh_grpc::error::{EnvError, EnvErrorCode};
use rlmesh_grpc::wire::{
    decode_batched_partial_values, encode_batched_partial_values, meta_map_from_proto,
    meta_map_to_proto, render_result_to_proto,
};
use rlmesh_proto::Edition;

use super::lanes::{LaneEnv, batch_infos, fold_phases};
use super::types::{
    CloseRequest, EpisodeMetadata, RenderRequest, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
};
use super::{Env, VectorEnv};
use crate::spaces;
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
    /// `(kind, path)` already reported this session. Behind a lock because
    /// lane ops run concurrently through `&self`.
    warned: std::sync::Mutex<HashSet<(String, String)>>,
    /// The edition the current session runs at (see
    /// [`Environment::pin_workflow_edition`]); the warning key is its promise.
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

    fn pin(&self, edition: Edition) {
        *self
            .edition
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = edition;
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
                    .insert((kind.to_string(), path.clone()));
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
        warnings: &mut Vec<ConformanceWarning>,
    ) -> Result<(), EnvError> {
        for value in values {
            self.enforce(space, value, kind, warnings)?;
        }
        Ok(())
    }
}

/// Encode a step's per-lane results into the wire reply, checking every
/// per-lane vector is `width` long.
fn encode_step_response(
    conformance: &Conformance,
    observation_space: &spaces::SpaceSpec,
    mut result: super::VectorStepResult,
    width: usize,
    mut warnings: Vec<ConformanceWarning>,
    env_indices: Vec<u32>,
) -> Result<ProtoStepResponse, EnvError> {
    validate_count(&result.observations, width, "observations")?;
    validate_count(&result.terminated, width, "terminated values")?;
    validate_count(&result.truncated, width, "truncated values")?;
    validate_count(&result.rewards, width, "rewards values")?;
    conformance.enforce_all(
        observation_space,
        &result.observations,
        "observation",
        &mut warnings,
    )?;
    inject_conformance_warnings(conformance.edition(), &mut result.info, warnings);
    let observations = encode_batched_partial_values(&result.observations, observation_space)
        .map_err(protocol_error_to_env_error)?;
    Ok(ProtoStepResponse {
        observation: Some(observations),
        rewards: result.rewards,
        terminated_mask: result.terminated.into_iter().map(u8::from).collect(),
        truncated_mask: result.truncated.into_iter().map(u8::from).collect(),
        infos: result.info.as_ref().map(meta_map_to_proto),
        completed_episodes: result
            .completed_episodes
            .iter()
            .map(public_episode_metadata_to_proto)
            .collect::<Result<Vec<_>, _>>()?,
        env_indices,
    })
}

/// Encode a reset's per-lane observations into the wire reply.
fn encode_reset_response(
    conformance: &Conformance,
    observation_space: &spaces::SpaceSpec,
    mut result: VectorResetResult,
    width: usize,
) -> Result<ProtoResetResponse, EnvError> {
    validate_count(&result.observations, width, "observations")?;
    let mut warnings = Vec::new();
    conformance.enforce_all(
        observation_space,
        &result.observations,
        "observation",
        &mut warnings,
    )?;
    inject_conformance_warnings(conformance.edition(), &mut result.info, warnings);
    let observations = encode_batched_partial_values(&result.observations, observation_space)
        .map_err(protocol_error_to_env_error)?;
    Ok(ProtoResetResponse {
        observation: Some(observations),
        infos: result.info.as_ref().map(meta_map_to_proto),
    })
}

/// Decode a wire action slab into `width` per-lane actions, conformance-checked.
fn decode_actions(
    conformance: &Conformance,
    action_space: &spaces::SpaceSpec,
    action: Option<&rlmesh_proto::spaces::v1::SpaceValue>,
    width: usize,
    warnings: &mut Vec<ConformanceWarning>,
) -> Result<Vec<spaces::SpaceValue>, EnvError> {
    // N is authoritative; a wrong-width/count action is a client fault, so a
    // decode failure maps to InvalidAction (not Internal).
    let actions = decode_batched_partial_values(action, action_space, width)
        .map_err(|err| EnvError::new(EnvErrorCode::InvalidAction, err.to_string()))?;
    validate_action_count(&actions, width)?;
    conformance.enforce_all(action_space, &actions, "action", warnings)?;
    Ok(actions)
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
}

#[async_trait]
impl<E: VectorEnv> Environment for WireEnvAdapter<E> {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        &self.observation_space
    }

    fn pin_workflow_edition(&self, edition: Edition) {
        self.conformance.pin(edition);
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        &self.action_space
    }

    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        &self.env_contract
    }

    async fn reset(
        &self,
        req: ProtoResetRequest,
    ) -> Result<(ProtoResetResponse, EndpointPhases), EnvError> {
        let decode_started = Instant::now();
        let request = VectorResetRequest {
            seeds: req.seeds,
            options: req.options.map(meta_map_from_proto),
            // Proto timeout_ms/env_indices are uint64/uint32; native is i64/i32.
            timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
            env_indices: proto_env_indices_to_native(req.env_indices),
        };
        // A reset naming lanes replies only those lanes; a whole-vector reset
        // replies full width.
        let width = if request.env_indices.is_empty() {
            self.num_envs
        } else {
            request.env_indices.len()
        };
        let partial = !request.env_indices.is_empty();
        let decode_ns = elapsed_ns(decode_started);

        let lock_started = Instant::now();
        let mut env = self.inner.lock().await;
        let queue_ns = elapsed_ns(lock_started);
        let call_started = Instant::now();
        // An env that cannot reset individual sub-envs inherits the rejecting
        // `reset_subset` default, so a partial reset fails loud here rather
        // than silently resetting the whole vector.
        let result = if partial {
            env.reset_subset(request).await
        } else {
            env.reset(request).await
        }
        .map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let inner = env.take_last_phases();
        drop(env);

        let encode_started = Instant::now();
        let response =
            encode_reset_response(&self.conformance, &self.observation_space, result, width)?;
        let mut phases =
            EndpointPhases::nest(decode_ns, call_ns, elapsed_ns(encode_started), inner);
        phases.queue_ns = queue_ns;
        Ok((response, phases))
    }

    async fn step(
        &self,
        req: ProtoStepRequest,
    ) -> Result<(ProtoStepResponse, EndpointPhases), EnvError> {
        if !req.env_indices.is_empty() {
            return Err(EnvError::new(
                EnvErrorCode::Unsupported,
                "this environment steps its whole vector in lockstep; a step naming lanes \
                 (StepRequest.env_indices) needs a lane endpoint",
            ));
        }
        let decode_started = Instant::now();
        let mut warnings = Vec::new();
        let actions = decode_actions(
            &self.conformance,
            &self.action_space,
            req.action.as_ref(),
            self.num_envs,
            &mut warnings,
        )?;
        let decode_ns = elapsed_ns(decode_started);

        let lock_started = Instant::now();
        let mut env = self.inner.lock().await;
        let queue_ns = elapsed_ns(lock_started);
        let call_started = Instant::now();
        let result = env
            .step(VectorStepRequest {
                actions,
                // Proto timeout_ms is uint64; native is i64.
                timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
            })
            .await
            .map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let inner = env.take_last_phases();
        drop(env);

        let encode_started = Instant::now();
        let response = encode_step_response(
            &self.conformance,
            &self.observation_space,
            result,
            self.num_envs,
            warnings,
            Vec::new(),
        )?;
        let mut phases =
            EndpointPhases::nest(decode_ns, call_ns, elapsed_ns(encode_started), inner);
        phases.queue_ns = queue_ns;
        Ok((response, phases))
    }

    async fn render(
        &self,
        req: ProtoRenderRequest,
    ) -> Result<(ProtoRenderResponse, EndpointPhases), EnvError> {
        let request = RenderRequest {
            env_index: render_env_index(&req.env_indices)?,
            // Proto timeout_ms is uint64; native is i64.
            timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
        };
        let lock_started = Instant::now();
        let mut env = self.inner.lock().await;
        let queue_ns = elapsed_ns(lock_started);
        let call_started = Instant::now();
        let result = env.render(request).await.map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let inner = env.take_last_phases();
        drop(env);

        let encode_started = Instant::now();
        let response = render_result_to_proto(&result);
        let mut phases = EndpointPhases::nest(0, call_ns, elapsed_ns(encode_started), inner);
        phases.queue_ns = queue_ns;
        Ok((response, phases))
    }

    async fn close(&self) -> Result<ProtoCloseResponse, EnvError> {
        let result = self
            .inner
            .lock()
            .await
            .close(CloseRequest {
                wait_for_episodes: false,
            })
            .await
            .map_err(gym_error_to_env_error)?;
        Ok(ProtoCloseResponse {
            final_episodes: result
                .final_episodes
                .iter()
                .map(public_episode_metadata_to_proto)
                .collect::<Result<Vec<_>, _>>()?,
        })
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

    /// The lanes a request covers: the ones it names, else every lane.
    fn lanes_for(&self, env_indices: &[u32]) -> Vec<usize> {
        if env_indices.is_empty() {
            (0..self.lanes.num_envs()).collect()
        } else {
            env_indices.iter().map(|&lane| lane as usize).collect()
        }
    }
}

#[async_trait]
impl Environment for WireLaneAdapter {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        self.lanes.observation_space()
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        self.lanes.action_space()
    }

    fn num_envs(&self) -> usize {
        self.lanes.num_envs()
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        self.lanes.env_contract()
    }

    fn supports_lanes(&self) -> bool {
        true
    }

    fn pin_workflow_edition(&self, edition: Edition) {
        self.conformance.pin(edition);
    }

    async fn reset(
        &self,
        req: ProtoResetRequest,
    ) -> Result<(ProtoResetResponse, EndpointPhases), EnvError> {
        let decode_started = Instant::now();
        let lanes = self.lanes_for(&req.env_indices);
        let options = req.options.map(meta_map_from_proto);
        let timeout_ms = i64::try_from(req.timeout_ms).unwrap_or(i64::MAX);
        let decode_ns = elapsed_ns(decode_started);

        // Each scalar lane gets its own slice of a per-lane option list, so a
        // list that does not cover the lanes fails the whole reset here rather
        // than per lane.
        let lane_options = (0..lanes.len())
            .map(|pos| lane_reset_options(options.as_ref(), pos, lanes.len()))
            .collect::<Result<Vec<_>, EnvError>>()?;

        let call_started = Instant::now();
        let results = futures::future::join_all(lanes.iter().enumerate().map(|(pos, &lane)| {
            let request = spaces::request::ResetRequest {
                seed: req.seeds.get(pos).copied(),
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
        for result in results {
            let (result, lane_phases) = result.map_err(gym_error_to_env_error)?;
            observations.extend(result.observation);
            infos.push(result.info);
            phases.push(lane_phases);
        }

        let encode_started = Instant::now();
        let response = encode_reset_response(
            &self.conformance,
            self.lanes.observation_space(),
            VectorResetResult {
                observations,
                info: batch_infos(infos, None),
                episode_ids: Vec::new(),
            },
            lanes.len(),
        )?;
        let inner = fold_phases(phases);
        let mut folded =
            EndpointPhases::nest(decode_ns, call_ns, elapsed_ns(encode_started), inner);
        folded.queue_ns = inner.queue_ns;
        Ok((response, folded))
    }

    async fn step(
        &self,
        req: ProtoStepRequest,
    ) -> Result<(ProtoStepResponse, EndpointPhases), EnvError> {
        let decode_started = Instant::now();
        let lanes = self.lanes_for(&req.env_indices);
        let mut warnings = Vec::new();
        let actions = decode_actions(
            &self.conformance,
            self.lanes.action_space(),
            req.action.as_ref(),
            lanes.len(),
            &mut warnings,
        )?;
        let timeout_ms = i64::try_from(req.timeout_ms).unwrap_or(i64::MAX);
        let decode_ns = elapsed_ns(decode_started);

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
        let mut result = super::VectorStepResult::default();
        let mut infos = Vec::with_capacity(lanes.len());
        let mut done = Vec::with_capacity(lanes.len());
        let mut phases = Vec::with_capacity(lanes.len());
        for lane_result in results {
            let (step, lane_phases) = lane_result.map_err(gym_error_to_env_error)?;
            result.observations.extend(step.observation);
            result.rewards.push(step.reward);
            result.terminated.push(step.terminated);
            result.truncated.push(step.truncated);
            done.push(step.terminated || step.truncated);
            infos.push(step.info);
            phases.push(lane_phases);
        }
        result.info = batch_infos(infos, Some(&done));

        let encode_started = Instant::now();
        let response = encode_step_response(
            &self.conformance,
            self.lanes.observation_space(),
            result,
            lanes.len(),
            warnings,
            req.env_indices,
        )?;
        let inner = fold_phases(phases);
        let mut folded =
            EndpointPhases::nest(decode_ns, call_ns, elapsed_ns(encode_started), inner);
        folded.queue_ns = inner.queue_ns;
        Ok((response, folded))
    }

    async fn render(
        &self,
        req: ProtoRenderRequest,
    ) -> Result<(ProtoRenderResponse, EndpointPhases), EnvError> {
        let lane = render_env_index(&req.env_indices)?.unwrap_or(0);
        let request = RenderRequest {
            env_index: Some(lane),
            timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
        };
        let call_started = Instant::now();
        let (result, inner) = self
            .lanes
            .render(lane, request)
            .await
            .map_err(gym_error_to_env_error)?;
        let call_ns = elapsed_ns(call_started);
        let encode_started = Instant::now();
        let response = render_result_to_proto(&result);
        let mut phases = EndpointPhases::nest(0, call_ns, elapsed_ns(encode_started), inner);
        phases.queue_ns = inner.queue_ns;
        Ok((response, phases))
    }

    async fn close(&self) -> Result<ProtoCloseResponse, EnvError> {
        self.lanes
            .close(CloseRequest {
                wait_for_episodes: false,
            })
            .await
            .map_err(gym_error_to_env_error)?;
        Ok(ProtoCloseResponse {
            final_episodes: Vec::new(),
        })
    }
}

fn gym_error_to_env_error(error: spaces::EnvRuntimeError) -> EnvError {
    match error {
        spaces::EnvRuntimeError::InvalidSpace(message)
        | spaces::EnvRuntimeError::InvalidValue(message) => {
            EnvError::new(EnvErrorCode::InvalidAction, message)
        }
        spaces::EnvRuntimeError::Runtime(message) => EnvError::new(EnvErrorCode::Internal, message),
        // EnvRuntimeError is #[non_exhaustive]; treat unknown variants as internal.
        other => EnvError::new(EnvErrorCode::Internal, other.to_string()),
    }
}

fn public_episode_metadata_to_proto(
    value: &EpisodeMetadata,
) -> std::result::Result<rlmesh_proto::env::v1::EpisodeMetadata, EnvError> {
    Ok(rlmesh_proto::env::v1::EpisodeMetadata {
        episode_id: value.episode_id.clone(),
        seed: value.seed,
        // Native env_index is i32 (>=0 lane offset); proto field is uint32.
        env_index: value.env_index.max(0) as u32,
        step_count: value.step_count,
        cumulative_reward: value.cumulative_reward,
        terminated: value.terminated,
        truncated: value.truncated,
        start_timestamp_ns: value.start_timestamp_ns,
        end_timestamp_ns: value.end_timestamp_ns,
        final_info: value.final_info.as_ref().map(meta_map_to_proto),
    })
}

pub(super) fn protocol_error_to_error(error: impl ToString) -> crate::Error {
    crate::Error::Internal(error.to_string())
}

pub(super) fn proto_episode_metadata_to_public(
    value: rlmesh_proto::env::v1::EpisodeMetadata,
) -> std::result::Result<EpisodeMetadata, String> {
    Ok(EpisodeMetadata {
        episode_id: value.episode_id,
        seed: value.seed,
        // Proto env_index is uint32; native field is i32 (lane offsets fit i32).
        env_index: i32::try_from(value.env_index).unwrap_or(i32::MAX),
        step_count: value.step_count,
        cumulative_reward: value.cumulative_reward,
        terminated: value.terminated,
        truncated: value.truncated,
        start_timestamp_ns: value.start_timestamp_ns,
        end_timestamp_ns: value.end_timestamp_ns,
        duration_ms: (value.end_timestamp_ns - value.start_timestamp_ns).max(0) / 1_000_000,
        final_info: value.final_info.map(meta_map_from_proto),
    })
}

/// Convert wire lane indices (uint32) to the native i32 representation. Lane
/// offsets always fit i32; an unrepresentable value is clamped rather than
/// wrapped so a foreign index is rejected loudly downstream.
fn proto_env_indices_to_native(env_indices: Vec<u32>) -> Vec<i32> {
    env_indices
        .into_iter()
        .map(|index| i32::try_from(index).unwrap_or(i32::MAX))
        .collect()
}

fn render_env_index(env_indices: &[u32]) -> std::result::Result<Option<usize>, EnvError> {
    match env_indices {
        [] => Ok(None),
        [index] => Ok(Some(*index as usize)),
        _ => Err(EnvError::new(
            EnvErrorCode::Unsupported,
            "render requests support at most one env_index".to_string(),
        )),
    }
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

fn protocol_error_to_env_error(error: impl ToString) -> EnvError {
    EnvError::new(EnvErrorCode::Internal, error.to_string())
}

/// The reset options lane `pos` of a `lanes`-wide fan-out receives.
///
/// The runtime sends the reserved `trial_index` option as a list in lane order
/// when a reset restarts several lanes (see
/// [`rlmesh_runtime::reset_options_for`]), but each lane here is a scalar env
/// that gets its own `reset`: handing it the whole list leaves
/// `rlmesh.trial_index(options)` unable to read an ordinal. Narrow the list to
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
    let Some(spaces::MetaValue::List(trials)) = options.get(rlmesh_runtime::TRIAL_INDEX_OPTION)
    else {
        return Ok(Some(options.clone()));
    };
    let Some(trial) = trials.get(pos) else {
        return Err(EnvError::new(
            EnvErrorCode::InvalidAction,
            format!(
                "reset option `{}` is a list of {} entries but this reset covers {lanes} lanes; \
                 send one ordinal per lane, in lane order",
                rlmesh_runtime::TRIAL_INDEX_OPTION,
                trials.len(),
            ),
        ));
    };
    let mut sliced = options.clone();
    sliced.insert(
        rlmesh_runtime::TRIAL_INDEX_OPTION.to_string(),
        trial.clone(),
    );
    Ok(Some(sliced))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crate::{BindAddress, Result, ServeOptions};

    use super::super::{
        RemoteVectorEnv as RemoteEnv, RenderResult, VectorCloseResult as CloseResult, VectorEnv,
        VectorEnvServer as EnvServer, VectorResetRequest as ResetRequest,
        VectorResetResult as ResetResult, VectorStepRequest as StepRequest,
        VectorStepResult as StepResult,
    };

    struct DummyEnv {
        obs_space: spaces::SpaceSpec,
        action_space: spaces::SpaceSpec,
        env_contract: spaces::EnvContract,
        last_render_request: Option<RenderRequest>,
        closes: Option<Arc<AtomicUsize>>,
        // The split this env reports for its own work, as a Python-backed env does.
        phases: EndpointPhases,
        // Real time each step spends inside the env, so a reported split is
        // contained in a measured call regardless of build profile.
        step_delay: Duration,
    }

    impl DummyEnv {
        fn new() -> Self {
            Self::new_with_close_counter(None)
        }

        fn new_with_close_counter(closes: Option<Arc<AtomicUsize>>) -> Self {
            let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2])
                .dtype(spaces::DType::Float32)
                .build()
                .unwrap();
            let action_space = spaces::spaces::DiscreteBuilder::new(3).build().unwrap();
            let env_contract = spaces::EnvContract {
                id: "DummyEnv-v1".to_string(),
                autoreset_mode: Default::default(),
                observation_space: Some(obs_space.clone()),
                action_space: Some(action_space.clone()),
                metadata: None,
                render_mode: String::new(),
                num_envs: 2,
            };

            Self {
                obs_space,
                action_space,
                env_contract,
                last_render_request: None,
                closes,
                phases: EndpointPhases::default(),
                step_delay: Duration::ZERO,
            }
        }
    }

    #[async_trait]
    impl VectorEnv for DummyEnv {
        fn observation_space(&self) -> &spaces::SpaceSpec {
            &self.obs_space
        }

        fn action_space(&self) -> &spaces::SpaceSpec {
            &self.action_space
        }

        fn num_envs(&self) -> usize {
            2
        }

        fn env_contract(&self) -> &spaces::EnvContract {
            &self.env_contract
        }

        async fn reset(
            &mut self,
            _req: ResetRequest,
        ) -> std::result::Result<ResetResult, spaces::EnvRuntimeError> {
            Ok(ResetResult {
                observations: vec![
                    spaces::SpaceValue::Box(
                        spaces::Tensor::from_vec(vec![0; 8], vec![2], spaces::DType::Float32)
                            .unwrap(),
                    ),
                    spaces::SpaceValue::Box(
                        spaces::Tensor::from_vec(vec![1; 8], vec![2], spaces::DType::Float32)
                            .unwrap(),
                    ),
                ],
                info: None,
                episode_ids: vec!["ep-0".to_string(), "ep-1".to_string()],
            })
        }

        async fn step(
            &mut self,
            req: StepRequest,
        ) -> std::result::Result<StepResult, spaces::EnvRuntimeError> {
            std::thread::sleep(self.step_delay);
            Ok(StepResult {
                observations: req
                    .actions
                    .into_iter()
                    .map(|action| match action {
                        spaces::SpaceValue::Discrete(value) => spaces::SpaceValue::Box(
                            spaces::Tensor::from_vec(
                                vec![value as u8; 8],
                                vec![2],
                                spaces::DType::Float32,
                            )
                            .unwrap(),
                        ),
                        other => other,
                    })
                    .collect(),
                rewards: vec![1.0, 2.0],
                terminated: vec![false, true],
                truncated: vec![false, false],
                info: None,
                completed_episodes: vec![],
                episode_ids: vec!["ep-0".to_string(), "ep-1b".to_string()],
            })
        }

        async fn render(
            &mut self,
            req: RenderRequest,
        ) -> std::result::Result<RenderResult, spaces::EnvRuntimeError> {
            self.last_render_request = Some(req);
            Ok(RenderResult {
                frame: Some(spaces::RenderFrame {
                    frame: vec![1, 2, 3],
                }),
            })
        }

        async fn close(
            &mut self,
            _req: CloseRequest,
        ) -> std::result::Result<CloseResult, spaces::EnvRuntimeError> {
            if let Some(closes) = &self.closes {
                closes.fetch_add(1, Ordering::SeqCst);
            }
            Ok(CloseResult {
                final_episodes: vec![],
            })
        }

        fn take_last_phases(&mut self) -> EndpointPhases {
            std::mem::take(&mut self.phases)
        }
    }

    fn step_request(adapter: &WireEnvAdapter<DummyEnv>) -> ProtoStepRequest {
        let actions = vec![
            spaces::SpaceValue::Discrete(1),
            spaces::SpaceValue::Discrete(2),
        ];
        ProtoStepRequest {
            action: Some(encode_batched_partial_values(&actions, adapter.action_space()).unwrap()),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn step_reports_the_wire_split_of_an_env_that_measures_nothing() {
        let adapter = WireEnvAdapter::new(DummyEnv::new());
        let request = step_request(&adapter);
        let (_, phases) = adapter.step(request).await.unwrap();

        assert!(phases.decode_ns > 0, "wire decode is measured");
        assert!(phases.encode_ns > 0, "wire encode is measured");
        // Nothing inside reported a split, so the whole call reads as user time.
        assert!(phases.user_ns > 0);
    }

    #[tokio::test]
    async fn step_folds_the_envs_own_split_into_the_wire_one() {
        let mut env = DummyEnv::new();
        env.phases = EndpointPhases {
            decode_ns: 100,
            user_ns: 5_000,
            encode_ns: 200,
            ..EndpointPhases::default()
        };
        env.step_delay = Duration::from_millis(1);
        let adapter = WireEnvAdapter::new(env);
        let request = step_request(&adapter);
        let (_, phases) = adapter.step(request).await.unwrap();

        // User time is the measured call (>= the 1ms the env really spent) minus
        // the env's own wire share, so it keeps the env's report plus the real
        // cost of reaching it; the env's conversion costs join the wire codec on
        // either side.
        assert!(phases.user_ns >= 1_000_000 - 300);
        assert!(phases.decode_ns > 100);
        assert!(phases.encode_ns > 200);
    }

    #[test]
    fn conformance_warnings_land_under_the_pinned_editions_key() {
        let conformance = Conformance {
            policy: ValidationPolicy::Warn,
            warned: Default::default(),
            edition: std::sync::Mutex::new(Edition::current()),
        };
        conformance.pin(Edition::E2026_06);
        let env = DummyEnv::new();
        let out_of_range = spaces::SpaceValue::Box(
            spaces::Tensor::from_vec(
                5.0f32.to_le_bytes().repeat(2),
                vec![2],
                spaces::DType::Float32,
            )
            .unwrap(),
        );
        let mut warnings = Vec::new();
        conformance
            .enforce(&env.obs_space, &out_of_range, "observation", &mut warnings)
            .expect("a range deviation warns under the warn policy");
        let mut info = None;
        inject_conformance_warnings(conformance.edition(), &mut info, warnings);
        assert!(
            info.unwrap().contains_key("rlmesh.conformance.warning"),
            "2026.06 reports conformance warnings under its reserved key"
        );
    }

    /// Reserve an ephemeral TCP port and free it, so a server can rebind it.
    async fn reserve_port() -> u16 {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    }

    /// Connect to a starting env server, retrying until it is listening.
    async fn connect_with_retry(
        address: &str,
        server: &tokio::task::JoinHandle<Result<()>>,
    ) -> RemoteEnv {
        loop {
            match RemoteEnv::connect(address).await {
                Ok(client) => break client,
                Err(err) if !server.is_finished() => {
                    let _ = err;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(err) => panic!("environment server did not start: {err}"),
            }
        }
    }

    /// Await a serve task that should exit cleanly after a shutdown request.
    async fn shutdown_and_join(server: tokio::task::JoinHandle<Result<()>>) {
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn wire_adapter_roundtrips_batched_reset_and_step() {
        let env = WireEnvAdapter::new(DummyEnv::new());

        let (reset, _) = Environment::reset(
            &env,
            ProtoResetRequest {
                seeds: vec![7, 8],
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let reset_obs =
            decode_batched_partial_values(reset.observation.as_ref(), env.observation_space(), 2)
                .unwrap();
        assert_eq!(reset_obs.len(), 2);

        let actions = [
            spaces::SpaceValue::Discrete(1),
            spaces::SpaceValue::Discrete(2),
        ];
        let action_space = env.action_space().clone();
        let (step, _) = Environment::step(
            &env,
            ProtoStepRequest {
                action: Some(encode_batched_partial_values(&actions, &action_space).unwrap()),
                timeout_ms: 0,
                env_indices: vec![],
                episode_ids: vec![],
            },
        )
        .await
        .unwrap();

        let step_obs =
            decode_batched_partial_values(step.observation.as_ref(), env.observation_space(), 2)
                .unwrap();
        assert_eq!(step_obs.len(), 2);
        assert_eq!(step.rewards, vec![1.0, 2.0]);
        assert_eq!(step.terminated_mask, vec![0, 1]);
    }

    #[tokio::test]
    async fn wire_adapter_rejects_wrong_action_count() {
        let env = WireEnvAdapter::new(DummyEnv::new());
        let actions = [spaces::SpaceValue::Discrete(1)];
        let action_space = env.action_space().clone();

        let error = Environment::step(
            &env,
            ProtoStepRequest {
                action: Some(encode_batched_partial_values(&actions, &action_space).unwrap()),
                timeout_ms: 0,
                env_indices: vec![],
                episode_ids: vec![],
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error.code, EnvErrorCode::InvalidAction);
    }

    #[tokio::test]
    async fn wire_adapter_maps_render_env_indices_to_env_index() {
        let env = WireEnvAdapter::new(DummyEnv::new());

        let (result, _) = Environment::render(
            &env,
            ProtoRenderRequest {
                env_indices: vec![1],
                timeout_ms: 0,
            },
        )
        .await
        .unwrap();

        assert!(result.frame.is_some());
    }

    #[tokio::test]
    async fn served_env_close_detaches_and_shutdown_stops_server() {
        let port = reserve_port().await;

        let closes = Arc::new(AtomicUsize::new(0));
        let server_closes = Arc::clone(&closes);
        let server = tokio::spawn(async move {
            EnvServer::new(DummyEnv::new_with_close_counter(Some(server_closes)))
                .serve_with_options(
                    BindAddress::Tcp {
                        host: "127.0.0.1".to_string(),
                        port,
                    },
                    ServeOptions {
                        allow_remote_shutdown: true,
                        ..ServeOptions::default()
                    },
                )
                .await
        });

        let address = format!("tcp://127.0.0.1:{port}");
        let mut client = connect_with_retry(&address, &server).await;

        let _ = client
            .reset(ResetRequest {
                seeds: vec![11, 22],
                ..ResetRequest::default()
            })
            .await
            .unwrap();
        let mut final_episodes = client.close().await.unwrap().final_episodes;
        final_episodes.sort_by_key(|episode| episode.env_index);
        assert_eq!(final_episodes.len(), 2);
        assert_eq!(final_episodes[0].env_index, 0);
        assert_eq!(final_episodes[0].seed, Some(11));
        assert_eq!(final_episodes[1].env_index, 1);
        assert_eq!(final_episodes[1].seed, Some(22));
        assert_eq!(closes.load(Ordering::SeqCst), 0);

        let mut second_client = RemoteEnv::connect(&address).await.unwrap();
        let _ = second_client.reset(ResetRequest::default()).await.unwrap();
        assert!(second_client.shutdown("test shutdown").await.unwrap());

        shutdown_and_join(server).await;

        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn detached_session_episodes_do_not_bleed_into_the_next_session() {
        let port = reserve_port().await;

        let server = tokio::spawn(async move {
            EnvServer::new(DummyEnv::new())
                .serve_with_options(
                    BindAddress::Tcp {
                        host: "127.0.0.1".to_string(),
                        port,
                    },
                    ServeOptions {
                        allow_remote_shutdown: true,
                        ..ServeOptions::default()
                    },
                )
                .await
        });

        let address = format!("tcp://127.0.0.1:{port}");
        let mut first = connect_with_retry(&address, &server).await;

        // Start episodes, then abandon the session without a graceful Close.
        let _ = first
            .reset(ResetRequest {
                seeds: vec![77, 88],
                ..ResetRequest::default()
            })
            .await
            .unwrap();
        first.detach();
        drop(first);

        // The slot frees once the server observes the stream end; retry until
        // the second session is admitted.
        let mut second = loop {
            let mut candidate = RemoteEnv::connect(&address).await.unwrap();
            match candidate.reset(ResetRequest::default()).await {
                Ok(_) => break candidate,
                Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        };

        let step = second
            .step(StepRequest {
                actions: vec![
                    spaces::SpaceValue::Discrete(0),
                    spaces::SpaceValue::Discrete(1),
                ],
                ..StepRequest::default()
            })
            .await
            .unwrap();
        for episode in &step.completed_episodes {
            assert_ne!(episode.seed, Some(77), "stale episode bled across sessions");
            assert_ne!(episode.seed, Some(88), "stale episode bled across sessions");
        }

        assert!(second.shutdown("test shutdown").await.unwrap());
        shutdown_and_join(server).await;
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn served_env_unix_socket_recovers_from_stale_socket_file() {
        let dir = std::env::temp_dir().join(format!("rlmesh-env-stale-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let socket_path = dir.join("env.sock");
        let _ = std::fs::remove_file(&socket_path);

        // Leave a stale socket file behind, as a previous unclean run would.
        // Without stale-socket cleanup, bind(2) returns AddrInUse forever.
        let stale = tokio::net::UnixListener::bind(&socket_path).unwrap();
        drop(stale);
        assert!(socket_path.exists(), "stale socket file must exist");

        let addr = BindAddress::Unix {
            path: socket_path.clone(),
        };
        let server = tokio::spawn({
            let addr = addr.clone();
            async move {
                EnvServer::new(DummyEnv::new())
                    .serve_with_options(
                        addr,
                        ServeOptions {
                            allow_remote_shutdown: true,
                            ..ServeOptions::default()
                        },
                    )
                    .await
            }
        });

        let address = format!("unix://{}", socket_path.display());
        let mut client = connect_with_retry(&address, &server).await;

        assert!(client.shutdown("test shutdown").await.unwrap());
        shutdown_and_join(server).await;

        // The socket file is unlinked after shutdown so a re-serve would succeed.
        assert!(
            !socket_path.exists(),
            "socket file must be unlinked after shutdown"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn env_serve_options_token_is_enforced_by_the_server() {
        let port = reserve_port().await;

        let addr = BindAddress::Tcp {
            host: "127.0.0.1".to_string(),
            port,
        };
        let server = tokio::spawn({
            let addr = addr.clone();
            async move {
                EnvServer::new(DummyEnv::new())
                    .serve_with_options(
                        addr,
                        ServeOptions {
                            allow_remote_shutdown: true,
                            token: Some("s3cret".to_string()),
                            ..ServeOptions::default()
                        },
                    )
                    .await
            }
        });

        let address = format!("tcp://127.0.0.1:{port}");

        // An unauthenticated facade connect is rejected: the token set through
        // ServeOptions actually reaches and is enforced by the env service.
        let connect_error = loop {
            match RemoteEnv::connect(&address).await {
                Ok(_) => panic!("unauthenticated connect must be rejected when a token is set"),
                Err(err) if !server.is_finished() => {
                    let message = err.to_string();
                    if message.contains("invalid env token") {
                        break message;
                    }
                    // Not yet listening; retry.
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(err) => panic!("env server did not start: {err}"),
            }
        };
        assert!(connect_error.contains("invalid env token"));

        // A token-bearing facade client connects and handshakes successfully.
        let env = RemoteEnv::connect_with_token(&address, "s3cret")
            .await
            .expect("facade connect_with_token must reach a token-protected env");
        drop(env);

        // A token-bearing client handshakes successfully.
        let mut authed = rlmesh_grpc::EnvClient::connect_with_token(&address, "s3cret")
            .await
            .unwrap();
        authed.handshake().await.expect("authorized handshake");
        assert!(authed.shutdown("test shutdown").await.unwrap().accepted);

        shutdown_and_join(server).await;
    }

    #[tokio::test]
    async fn remote_env_reset_and_step_decode_with_shared_specs() {
        let bound = EnvServer::new(DummyEnv::new())
            .bind_with_options(
                BindAddress::Tcp {
                    host: "127.0.0.1".to_string(),
                    port: 0,
                },
                ServeOptions {
                    allow_remote_shutdown: true,
                    ..ServeOptions::default()
                },
            )
            .await
            .unwrap();
        let port = match bound.local_addr().clone() {
            BindAddress::Tcp { port, .. } => port,
            other => panic!("expected tcp, got {other:?}"),
        };
        let server = tokio::spawn(async move { bound.serve().await });

        let address = format!("tcp://127.0.0.1:{port}");
        let mut client = RemoteEnv::connect(&address).await.unwrap();

        // Multiple reset/step calls reuse the Arc-shared specs (no per-call deep
        // clone) and still decode the expected number of observations.
        for _ in 0..3 {
            let reset = client.reset(ResetRequest::default()).await.unwrap();
            assert_eq!(reset.observations.len(), client.num_envs());
            let step = client
                .step(VectorStepRequest {
                    actions: vec![
                        spaces::SpaceValue::Discrete(0),
                        spaces::SpaceValue::Discrete(1),
                    ],
                    timeout_ms: 0,
                })
                .await
                .unwrap();
            assert_eq!(step.observations.len(), client.num_envs());
        }

        assert!(client.shutdown("done").await.unwrap());
        shutdown_and_join(server).await;
    }

    #[tokio::test]
    async fn env_bind_resolves_port_zero_before_serving() {
        let bound = EnvServer::new(DummyEnv::new())
            .bind_with_options(
                BindAddress::Tcp {
                    host: "127.0.0.1".to_string(),
                    port: 0,
                },
                ServeOptions {
                    allow_remote_shutdown: true,
                    ..ServeOptions::default()
                },
            )
            .await
            .unwrap();

        // The OS-assigned port is known before we await shutdown.
        let resolved = bound.local_addr().clone();
        let port = match resolved {
            BindAddress::Tcp { port, .. } => port,
            other => panic!("expected tcp bind address, got {other:?}"),
        };
        assert_ne!(port, 0, "port 0 must resolve to a real port");

        let server = tokio::spawn(async move { bound.serve().await });

        // No poll-connect race: the resolved address is immediately usable.
        let address = format!("tcp://127.0.0.1:{port}");
        let mut client = RemoteEnv::connect(&address).await.unwrap();
        let _ = client.reset(ResetRequest::default()).await.unwrap();
        assert!(client.shutdown("test shutdown").await.unwrap());

        shutdown_and_join(server).await;
    }

    #[tokio::test]
    async fn served_env_reports_grpc_health_serving() {
        use tonic_health::ServingStatus;
        use tonic_health::pb::HealthCheckRequest;
        use tonic_health::pb::health_client::HealthClient;

        let bound = EnvServer::new(DummyEnv::new())
            .bind_with_options(
                BindAddress::Tcp {
                    host: "127.0.0.1".to_string(),
                    port: 0,
                },
                ServeOptions {
                    allow_remote_shutdown: true,
                    ..ServeOptions::default()
                },
            )
            .await
            .unwrap();
        let port = match bound.local_addr().clone() {
            BindAddress::Tcp { port, .. } => port,
            other => panic!("expected tcp, got {other:?}"),
        };
        let server = tokio::spawn(async move { bound.serve().await });

        // A standard grpc.health.v1 client sees overall server health = SERVING.
        let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut health = HealthClient::new(channel);
        let response = health
            .check(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(response.status, ServingStatus::Serving as i32);

        // Shut the server down through the existing env client path.
        let mut client = RemoteEnv::connect(&format!("tcp://127.0.0.1:{port}"))
            .await
            .unwrap();
        assert!(client.shutdown("done").await.unwrap());
        shutdown_and_join(server).await;
    }
    /// One lane of a [`WireLaneAdapter`] that records the reset options it saw.
    struct RecordingLane {
        obs_space: spaces::SpaceSpec,
        action_space: spaces::SpaceSpec,
        env_contract: spaces::EnvContract,
        seen: Arc<Mutex<Vec<Option<spaces::MetaMap>>>>,
    }

    impl RecordingLane {
        fn new(seen: Arc<Mutex<Vec<Option<spaces::MetaMap>>>>) -> Self {
            let obs_space = spaces::spaces::BoxSpaceBuilder::scalar(-1.0, 1.0, vec![2])
                .dtype(spaces::DType::Float32)
                .build()
                .unwrap();
            let action_space = spaces::spaces::DiscreteBuilder::new(3).build().unwrap();
            let env_contract = spaces::EnvContract {
                id: "RecordingLane-v1".to_string(),
                autoreset_mode: Default::default(),
                observation_space: Some(obs_space.clone()),
                action_space: Some(action_space.clone()),
                metadata: None,
                render_mode: String::new(),
                num_envs: 1,
            };
            Self {
                obs_space,
                action_space,
                env_contract,
                seen,
            }
        }

        fn observation() -> spaces::SpaceValue {
            spaces::SpaceValue::Box(
                spaces::Tensor::from_vec(vec![0; 8], vec![2], spaces::DType::Float32).unwrap(),
            )
        }
    }

    #[async_trait]
    impl super::super::Env for RecordingLane {
        fn observation_space(&self) -> &spaces::SpaceSpec {
            &self.obs_space
        }

        fn action_space(&self) -> &spaces::SpaceSpec {
            &self.action_space
        }

        fn env_contract(&self) -> &spaces::EnvContract {
            &self.env_contract
        }

        async fn reset(
            &mut self,
            req: spaces::request::ResetRequest,
        ) -> std::result::Result<spaces::request::ResetResult, spaces::EnvRuntimeError> {
            self.seen.lock().unwrap().push(req.options);
            Ok(spaces::request::ResetResult {
                observation: Some(Self::observation()),
                info: None,
                episode_id: None,
            })
        }

        async fn step(
            &mut self,
            _req: spaces::request::StepRequest,
        ) -> std::result::Result<spaces::request::StepResult, spaces::EnvRuntimeError> {
            Ok(spaces::request::StepResult {
                observation: Some(Self::observation()),
                ..Default::default()
            })
        }

        async fn render(
            &mut self,
            _req: RenderRequest,
        ) -> std::result::Result<RenderResult, spaces::EnvRuntimeError> {
            Ok(RenderResult { frame: None })
        }

        async fn close(
            &mut self,
            _req: spaces::CloseRequest,
        ) -> std::result::Result<spaces::request::CloseResult, spaces::EnvRuntimeError> {
            Ok(spaces::request::CloseResult)
        }
    }

    fn trial_list(trials: &[i64]) -> spaces::MetaMap {
        [(
            rlmesh_runtime::TRIAL_INDEX_OPTION.to_string(),
            spaces::MetaValue::List(trials.iter().copied().map(spaces::MetaValue::Int).collect()),
        )]
        .into_iter()
        .collect()
    }

    #[test]
    fn lane_reset_options_narrows_the_trial_list_to_one_lane() {
        let mut options = trial_list(&[7, 8, 9]);
        options.insert("other".to_string(), spaces::MetaValue::Bool(true));

        for (pos, expected) in [(0, 7), (1, 8), (2, 9)] {
            let sliced = lane_reset_options(Some(&options), pos, 3).unwrap().unwrap();
            assert_eq!(
                sliced.get(rlmesh_runtime::TRIAL_INDEX_OPTION),
                Some(&spaces::MetaValue::Int(expected)),
            );
            // Every other key rides through untouched.
            assert_eq!(
                sliced.get("other"),
                Some(&spaces::MetaValue::Bool(true)),
                "only trial_index is sliced",
            );
        }

        // A bare integer (the single-lane encoding) and a map without the key
        // are both passed on as they are.
        let scalar: spaces::MetaMap = [(
            rlmesh_runtime::TRIAL_INDEX_OPTION.to_string(),
            spaces::MetaValue::Int(4),
        )]
        .into_iter()
        .collect();
        assert_eq!(
            lane_reset_options(Some(&scalar), 0, 1).unwrap(),
            Some(scalar.clone()),
        );
        assert_eq!(lane_reset_options(None, 0, 1).unwrap(), None);
    }

    #[test]
    fn a_trial_list_shorter_than_the_lanes_is_rejected() {
        let error = lane_reset_options(Some(&trial_list(&[7])), 1, 2).unwrap_err();

        assert_eq!(error.code, EnvErrorCode::InvalidAction);
        assert!(
            error.message.contains("list of 1 entries") && error.message.contains("covers 2 lanes"),
            "the error must name both lengths, got {}",
            error.message,
        );
    }

    #[tokio::test]
    async fn a_lane_reset_fails_loud_on_a_short_trial_list() {
        let seen: Arc<Mutex<Vec<Option<spaces::MetaMap>>>> = Arc::new(Mutex::new(Vec::new()));
        let adapter = WireLaneAdapter::new(vec![
            RecordingLane::new(seen.clone()),
            RecordingLane::new(seen.clone()),
        ])
        .unwrap();

        let error = adapter
            .reset(ProtoResetRequest {
                seeds: vec![1, 2],
                options: Some(meta_map_to_proto(&trial_list(&[10]))),
                ..Default::default()
            })
            .await
            .unwrap_err();

        assert_eq!(error.code, EnvErrorCode::InvalidAction);
        // Rejected before any lane ran, so no lane reset on a half-read option.
        assert!(seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn lane_fan_out_gives_each_lane_its_own_trial_index() {
        // The runtime sends a multi-lane reset one list in lane order; each
        // scalar lane must see its own integer, the way `rlmesh.trial_index()`
        // reads it.
        let seen: Arc<Mutex<Vec<Option<spaces::MetaMap>>>> = Arc::new(Mutex::new(Vec::new()));
        let adapter = WireLaneAdapter::new(vec![
            RecordingLane::new(seen.clone()),
            RecordingLane::new(seen.clone()),
            RecordingLane::new(seen.clone()),
        ])
        .unwrap();

        adapter
            .reset(ProtoResetRequest {
                seeds: vec![1, 2, 3],
                options: Some(meta_map_to_proto(&trial_list(&[10, 11, 12]))),
                ..Default::default()
            })
            .await
            .unwrap();

        let mut trials = seen
            .lock()
            .unwrap()
            .iter()
            .map(|options| {
                options
                    .as_ref()
                    .and_then(|options| options.get(rlmesh_runtime::TRIAL_INDEX_OPTION).cloned())
            })
            .collect::<Vec<_>>();
        trials.sort_by_key(|trial| match trial {
            Some(spaces::MetaValue::Int(value)) => *value,
            other => panic!("expected an integer trial_index, got {other:?}"),
        });
        assert_eq!(
            trials,
            vec![
                Some(spaces::MetaValue::Int(10)),
                Some(spaces::MetaValue::Int(11)),
                Some(spaces::MetaValue::Int(12)),
            ],
        );
    }
}
