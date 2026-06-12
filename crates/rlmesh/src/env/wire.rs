use async_trait::async_trait;
use rlmesh_grpc::env::{
    CloseResponse as ProtoCloseResponse, Environment, RenderRequest as ProtoRenderRequest,
    RenderResponse as ProtoRenderResponse, ResetRequest as ProtoResetRequest,
    ResetResponse as ProtoResetResponse, StepRequest as ProtoStepRequest,
    StepResponse as ProtoStepResponse,
};
use rlmesh_grpc::error::{EnvError, EnvErrorCode};
use rlmesh_grpc::wire::{
    bytes_value, decode_batched_partial_values, encode_batched_partial_values,
    meta_map_from_struct, meta_map_to_struct, render_result_to_proto, value_bytes,
};

use super::Env;
use super::types::{CloseRequest, EpisodeMetadata, RenderRequest, ResetRequest, StepRequest};
use crate::spaces;

pub struct WireEnvAdapter<E> {
    inner: E,
}

impl<E> WireEnvAdapter<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<E: Env> Environment for WireEnvAdapter<E> {
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
            .reset(ResetRequest {
                seeds: req.seeds,
                options: req.options.map(meta_map_from_struct),
                timeout_ms: req.timeout_ms,
            })
            .await
            .map_err(gym_error_to_env_error)?;

        validate_observation_count(&result.observations, self.inner.num_envs())?;

        let observations =
            encode_batched_partial_values(&result.observations, self.inner.observation_space())
                .map_err(protocol_error_to_env_error)?;

        Ok(ProtoResetResponse {
            observation: Some(bytes_value(observations)),
            infos: result.info.as_ref().map(meta_map_to_struct),
            episode_ids: result.episode_ids,
        })
    }

    async fn step(
        &mut self,
        req: ProtoStepRequest,
    ) -> std::result::Result<ProtoStepResponse, EnvError> {
        let action_payload =
            value_bytes(req.action.as_ref()).map_err(protocol_error_to_env_error)?;
        let actions =
            decode_batched_partial_values(action_payload.as_ref(), self.inner.action_space())
                .map_err(protocol_error_to_env_error)?;
        validate_action_count(&actions, self.inner.num_envs())?;

        let result = self
            .inner
            .step(StepRequest {
                actions,
                timeout_ms: req.timeout_ms,
            })
            .await
            .map_err(gym_error_to_env_error)?;

        let env_count = self.inner.num_envs();
        validate_observation_count(&result.observations, env_count)?;
        validate_bool_count(&result.terminated, env_count, "terminated")?;
        validate_bool_count(&result.truncated, env_count, "truncated")?;
        validate_f64_count(&result.rewards, env_count, "rewards")?;

        let observations =
            encode_batched_partial_values(&result.observations, self.inner.observation_space())
                .map_err(protocol_error_to_env_error)?;

        Ok(ProtoStepResponse {
            observation: Some(bytes_value(observations)),
            rewards: result.rewards,
            terminated_mask: result.terminated.into_iter().map(u8::from).collect(),
            truncated_mask: result.truncated.into_iter().map(u8::from).collect(),
            infos: result.info.as_ref().map(meta_map_to_struct),
            completed_episodes: result
                .completed_episodes
                .iter()
                .map(public_episode_metadata_to_proto)
                .collect::<std::result::Result<Vec<_>, _>>()?,
            episode_ids: result.episode_ids,
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
                timeout_ms: req.timeout_ms,
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
    }
}

fn public_episode_metadata_to_proto(
    value: &EpisodeMetadata,
) -> std::result::Result<rlmesh_proto::env::v1::EpisodeMetadata, EnvError> {
    Ok(rlmesh_proto::env::v1::EpisodeMetadata {
        episode_id: value.episode_id.clone(),
        seed: value.seed,
        env_index: value.env_index,
        step_count: value.step_count,
        cumulative_reward: value.cumulative_reward,
        terminated: value.terminated,
        truncated: value.truncated,
        start_timestamp_ns: value.start_timestamp_ns,
        end_timestamp_ns: value.end_timestamp_ns,
        duration_ms: value.duration_ms,
        final_info: value.final_info.as_ref().map(meta_map_to_struct),
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
        env_index: value.env_index,
        step_count: value.step_count,
        cumulative_reward: value.cumulative_reward,
        terminated: value.terminated,
        truncated: value.truncated,
        start_timestamp_ns: value.start_timestamp_ns,
        end_timestamp_ns: value.end_timestamp_ns,
        duration_ms: value.duration_ms,
        final_info: value.final_info.map(meta_map_from_struct),
    })
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

pub(super) fn validate_observation_count(
    observations: &[spaces::SpaceValue],
    num_envs: usize,
) -> std::result::Result<(), EnvError> {
    if observations.len() == num_envs {
        return Ok(());
    }
    Err(EnvError::new(
        EnvErrorCode::Internal,
        format!(
            "env returned {} observations for {num_envs} envs",
            observations.len()
        ),
    ))
}

pub(super) fn validate_bool_count(
    values: &[bool],
    num_envs: usize,
    label: &str,
) -> std::result::Result<(), EnvError> {
    if values.len() == num_envs {
        return Ok(());
    }
    Err(EnvError::new(
        EnvErrorCode::Internal,
        format!(
            "env returned {} {label} values for {num_envs} envs",
            values.len()
        ),
    ))
}

pub(super) fn validate_f64_count(
    values: &[f64],
    num_envs: usize,
    label: &str,
) -> std::result::Result<(), EnvError> {
    if values.len() == num_envs {
        return Ok(());
    }
    Err(EnvError::new(
        EnvErrorCode::Internal,
        format!(
            "env returned {} {label} values for {num_envs} envs",
            values.len()
        ),
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

    use crate::{BindAddress, ServeOptions};

    use super::super::{
        CloseResult, EnvServer, RemoteEnv, RenderResult, ResetRequest, ResetResult, StepResult,
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
    impl Env for DummyEnv {
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
                    png_frame: vec![1, 2, 3],
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
        let reset_payload = value_bytes(reset.observation.as_ref()).unwrap().unwrap();
        let reset_obs =
            decode_batched_partial_values(Some(&reset_payload), env.observation_space()).unwrap();
        assert_eq!(reset_obs.len(), 2);

        let actions = [
            spaces::SpaceValue::Discrete(1),
            spaces::SpaceValue::Discrete(2),
        ];
        let action_space = env.action_space().clone();
        let step = Environment::step(
            &mut env,
            ProtoStepRequest {
                action: Some(bytes_value(
                    encode_batched_partial_values(&actions, &action_space).unwrap(),
                )),
                timeout_ms: 0,
            },
        )
        .await
        .unwrap();

        let step_payload = value_bytes(step.observation.as_ref()).unwrap().unwrap();
        let step_obs =
            decode_batched_partial_values(Some(&step_payload), env.observation_space()).unwrap();
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
                action: Some(bytes_value(
                    encode_batched_partial_values(&actions, &action_space).unwrap(),
                )),
                timeout_ms: 0,
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

        assert!(result.png_frame.is_some());
    }

    #[tokio::test]
    async fn served_env_close_detaches_and_shutdown_stops_server() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

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
        let mut client = loop {
            match RemoteEnv::connect(&address).await {
                Ok(client) => break client,
                Err(err) if !server.is_finished() => {
                    let _ = err;
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                Err(err) => panic!("environment server did not start: {err}"),
            }
        };

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
        assert_eq!(final_episodes[0].seed, 11);
        assert_eq!(final_episodes[1].env_index, 1);
        assert_eq!(final_episodes[1].seed, 22);
        assert_eq!(closes.load(Ordering::SeqCst), 0);

        let mut second_client = RemoteEnv::connect(&address).await.unwrap();
        let _ = second_client.reset(ResetRequest::default()).await.unwrap();
        assert!(second_client.shutdown("test shutdown").await.unwrap());

        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }
}
