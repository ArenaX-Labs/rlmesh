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

use super::types::{
    CloseRequest, EpisodeMetadata, RenderRequest, ResetRequest as VectorResetRequest,
    ResetResult as VectorResetResult, StepRequest as VectorStepRequest,
};
use super::{Env, VectorEnv};
use crate::spaces;
use rlmesh_spaces::spaces::{PolicyOutcome, ValidationPolicy};
use std::collections::{BTreeMap, HashSet};

/// Reserved info-map key carrying value-conformance warnings (2026.06 edition).
const CONFORMANCE_WARNING_KEY: &str = "rlmesh.conformance.warning";

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

/// Merge conformance warnings into an info map under the reserved key.
fn inject_conformance_warnings(
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
        CONFORMANCE_WARNING_KEY.to_string(),
        spaces::MetaValue::List(entries),
    );
}

/// Internal adapter bridging an [`Env`] to the gRPC `Environment` trait.
#[doc(hidden)]
pub struct WireEnvAdapter<E> {
    inner: E,
    /// Serving-side validation policy for observation/action range deviations.
    policy: ValidationPolicy,
    /// Conformance-warning dedup: `(kind, path)` already reported this session.
    warned: HashSet<(String, String)>,
}

/// Internal adapter bridging a scalar [`Env`] to the vectorized wire layer.
#[doc(hidden)]
pub struct ScalarEnvAdapter<E> {
    inner: E,
}

impl<E> ScalarEnvAdapter<E> {
    /// Wrap a scalar [`Env`] implementation.
    #[doc(hidden)]
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<E: Env> VectorEnv for ScalarEnvAdapter<E> {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        self.inner.observation_space()
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        self.inner.action_space()
    }

    fn num_envs(&self) -> usize {
        1
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        self.inner.env_contract()
    }

    async fn reset(
        &mut self,
        req: VectorResetRequest,
    ) -> std::result::Result<VectorResetResult, spaces::EnvRuntimeError> {
        let result = self
            .inner
            .reset(spaces::request::ResetRequest {
                seed: req.seeds.first().copied(),
                options: req.options,
                timeout_ms: req.timeout_ms,
            })
            .await?;

        Ok(VectorResetResult {
            observations: result.observation.into_iter().collect(),
            info: result.info,
            episode_ids: result.episode_id.into_iter().collect(),
        })
    }

    async fn step(
        &mut self,
        req: VectorStepRequest,
    ) -> std::result::Result<super::VectorStepResult, spaces::EnvRuntimeError> {
        let result = self
            .inner
            .step(spaces::request::StepRequest {
                action: req.actions.into_iter().next(),
                timeout_ms: req.timeout_ms,
            })
            .await?;

        Ok(super::VectorStepResult {
            observations: result.observation.into_iter().collect(),
            rewards: vec![result.reward],
            terminated: vec![result.terminated],
            truncated: vec![result.truncated],
            info: result.info,
            completed_episodes: vec![],
            episode_ids: vec![],
        })
    }

    async fn render(
        &mut self,
        req: RenderRequest,
    ) -> std::result::Result<spaces::RenderResult, spaces::EnvRuntimeError> {
        self.inner.render(req).await
    }

    async fn close(
        &mut self,
        req: CloseRequest,
    ) -> std::result::Result<super::VectorCloseResult, spaces::EnvRuntimeError> {
        let _ = self.inner.close(req).await?;
        Ok(super::VectorCloseResult {
            final_episodes: vec![],
        })
    }
}

impl<E> WireEnvAdapter<E> {
    /// Wrap a [`VectorEnv`] for the wire layer.
    #[doc(hidden)]
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            policy: validation_policy_from_env(),
            warned: HashSet::new(),
        }
    }
}

impl<E: VectorEnv> WireEnvAdapter<E> {
    /// Encode a public [`ResetResult`] into the proto reset response, validating
    /// the observation batch width. Shared by `reset` and `reset_subset`.
    fn encode_reset_response(
        &mut self,
        mut result: VectorResetResult,
    ) -> std::result::Result<ProtoResetResponse, EnvError> {
        validate_count(&result.observations, self.inner.num_envs(), "observations")?;

        let mut warnings = Vec::new();
        for observation in &result.observations {
            self.enforce(observation, "observation", &mut warnings)?;
        }
        inject_conformance_warnings(&mut result.info, warnings);

        let observations =
            encode_batched_partial_values(&result.observations, self.inner.observation_space())
                .map_err(protocol_error_to_env_error)?;

        Ok(ProtoResetResponse {
            observation: Some(observations),
            infos: result.info.as_ref().map(meta_map_to_proto),
            episode_ids: result.episode_ids,
        })
    }

    /// Validate one observation or action against its declared space under the
    /// active policy: structural deviations always reject, range deviations
    /// follow the policy (default `warn`, recorded once per `(kind, path)`).
    fn enforce(
        &mut self,
        value: &spaces::SpaceValue,
        kind: &str,
        warnings: &mut Vec<ConformanceWarning>,
    ) -> std::result::Result<(), EnvError> {
        let space = if kind == "action" {
            self.inner.action_space()
        } else {
            self.inner.observation_space()
        };
        let outcome = self.policy.check(space, value);
        match outcome {
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
                if self.warned.insert((kind.to_string(), path.clone())) {
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
}

#[async_trait]
impl<E: VectorEnv> Environment for WireEnvAdapter<E> {
    fn observation_space(&self) -> &spaces::SpaceSpec {
        self.inner.observation_space()
    }

    fn action_space(&self) -> &spaces::SpaceSpec {
        self.inner.action_space()
    }

    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }

    fn env_contract(&self) -> &spaces::EnvContract {
        self.inner.env_contract()
    }

    async fn reset(
        &mut self,
        req: ProtoResetRequest,
    ) -> std::result::Result<ProtoResetResponse, EnvError> {
        let result = self
            .inner
            .reset(VectorResetRequest {
                seeds: req.seeds,
                options: req.options.map(meta_map_from_proto),
                // Proto timeout_ms/env_indices are uint64/uint32; native is i64/i32.
                timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
                env_indices: proto_env_indices_to_native(req.env_indices),
            })
            .await
            .map_err(gym_error_to_env_error)?;

        self.encode_reset_response(result)
    }

    /// Partial / per-lane reset: forward the requested lane indices to the inner
    /// env's [`reset_subset`](Env::reset_subset). An env that cannot reset
    /// individual sub-envs inherits the rejecting default, so it fails loud here
    /// rather than silently resetting the whole vector.
    async fn reset_subset(
        &mut self,
        req: ProtoResetRequest,
    ) -> std::result::Result<ProtoResetResponse, EnvError> {
        let result = self
            .inner
            .reset_subset(VectorResetRequest {
                seeds: req.seeds,
                options: req.options.map(meta_map_from_proto),
                // Proto timeout_ms/env_indices are uint64/uint32; native is i64/i32.
                timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
                env_indices: proto_env_indices_to_native(req.env_indices),
            })
            .await
            .map_err(gym_error_to_env_error)?;

        self.encode_reset_response(result)
    }

    async fn step(
        &mut self,
        req: ProtoStepRequest,
    ) -> std::result::Result<ProtoStepResponse, EnvError> {
        // N is authoritative (num_envs); a wrong-width/count action is a client
        // fault, so a decode failure maps to InvalidAction (not Internal).
        let num_envs = self.inner.num_envs();
        let actions =
            decode_batched_partial_values(req.action.as_ref(), self.inner.action_space(), num_envs)
                .map_err(|err| EnvError::new(EnvErrorCode::InvalidAction, err.to_string()))?;
        validate_action_count(&actions, num_envs)?;

        let mut warnings = Vec::new();
        for action in &actions {
            self.enforce(action, "action", &mut warnings)?;
        }

        let mut result = self
            .inner
            .step(VectorStepRequest {
                actions,
                // Proto timeout_ms is uint64; native is i64.
                timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
            })
            .await
            .map_err(gym_error_to_env_error)?;

        let env_count = self.inner.num_envs();
        validate_count(&result.observations, env_count, "observations")?;
        validate_count(&result.terminated, env_count, "terminated values")?;
        validate_count(&result.truncated, env_count, "truncated values")?;
        validate_count(&result.rewards, env_count, "rewards values")?;

        for observation in &result.observations {
            self.enforce(observation, "observation", &mut warnings)?;
        }
        inject_conformance_warnings(&mut result.info, warnings);

        let observations =
            encode_batched_partial_values(&result.observations, self.inner.observation_space())
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
                .collect::<std::result::Result<Vec<_>, _>>()?,
            episode_ids: result.episode_ids,
            // Full-width response; partial-width is reserved-but-deferred.
            env_indices: Vec::new(),
        })
    }

    async fn render(
        &mut self,
        req: ProtoRenderRequest,
    ) -> std::result::Result<ProtoRenderResponse, EnvError> {
        let result = self
            .inner
            .render(RenderRequest {
                env_index: render_env_index(&req.mask)?,
                // Proto timeout_ms is uint64; native is i64.
                timeout_ms: i64::try_from(req.timeout_ms).unwrap_or(i64::MAX),
            })
            .await
            .map_err(gym_error_to_env_error)?;
        Ok(render_result_to_proto(&result))
    }

    async fn close(&mut self) -> std::result::Result<ProtoCloseResponse, EnvError> {
        let result = self
            .inner
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
                .collect::<std::result::Result<Vec<_>, _>>()?,
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
        // Native duration_ms is i64 (>=0); proto field is uint64.
        duration_ms: value.duration_ms.max(0) as u64,
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
        // Proto duration_ms is uint64; native field is i64.
        duration_ms: i64::try_from(value.duration_ms).unwrap_or(i64::MAX),
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

fn render_env_index(mask: &[u8]) -> std::result::Result<Option<usize>, EnvError> {
    let indices = mask
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value != 0).then_some(index))
        .collect::<Vec<_>>();

    match indices.as_slice() {
        [] => Ok(None),
        [index] => Ok(Some(*index)),
        _ => Err(EnvError::new(
            EnvErrorCode::InvalidAction,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
        let mut env = WireEnvAdapter::new(DummyEnv::new());

        let reset = Environment::reset(
            &mut env,
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
        let step = Environment::step(
            &mut env,
            ProtoStepRequest {
                action: Some(encode_batched_partial_values(&actions, &action_space).unwrap()),
                timeout_ms: 0,
                env_indices: vec![],
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
        let mut env = WireEnvAdapter::new(DummyEnv::new());
        let actions = [spaces::SpaceValue::Discrete(1)];
        let action_space = env.action_space().clone();

        let error = Environment::step(
            &mut env,
            ProtoStepRequest {
                action: Some(encode_batched_partial_values(&actions, &action_space).unwrap()),
                timeout_ms: 0,
                env_indices: vec![],
            },
        )
        .await
        .unwrap_err();

        assert_eq!(error.code, EnvErrorCode::InvalidAction);
    }

    #[tokio::test]
    async fn wire_adapter_maps_render_mask_to_env_index() {
        let mut env = WireEnvAdapter::new(DummyEnv::new());

        let result = Environment::render(
            &mut env,
            ProtoRenderRequest {
                mask: vec![0, 1],
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
}
