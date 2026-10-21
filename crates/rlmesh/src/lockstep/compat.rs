//! The Gym compatibility path (§12.4), driver side: [`LockstepEnv`] is a
//! [`RuntimeEnv`], so the v1 [`RuntimeDriver`](rlmesh_runtime::RuntimeDriver)
//! drives a lockstep env with its chunking, prefetch, batching, autoreset
//! handling and driver-minted episode ids unchanged. The lockstep reply
//! carries no episode metadata; an [`EpisodeMirror`] rebuilds the
//! completions the env server would have reported.
//!
//! v1 mints UUIDv7 episode ids and pushes a fresh one down on the step that
//! rolls a NEXT_STEP autoreset; v2 ids are `u64`. The driver numbers its
//! ids (1, 2, ...) for `LockstepReset.episode_ids` and keeps the v1 strings
//! itself.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use rlmesh_grpc::env::{EnvHandshake, EpisodeMirror};
use rlmesh_grpc::error::{Error as GrpcError, ProtocolError, TransportError};
use rlmesh_grpc::wire::{meta_map_from_proto, meta_map_to_proto};
use rlmesh_proto::env::v1::{
    RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest, StepResponse,
};
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_runtime::{RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError};
use rlmesh_session::wire::{ChannelId, Epoch, LeaseToken, LockstepReset, LockstepStep};

use crate::lockstep::error::GymError;
use crate::lockstep::exchange::{LockstepLink, Request};
use crate::lockstep::map::{self, GymEnvInfo, GymMap};
use crate::lockstep::meta_json;
use crate::lockstep::served::{ordered, tagged};

fn rewards_from(bytes: &[u8], lanes: usize) -> Result<Vec<f64>, GymError> {
    if bytes.len() == lanes * 8 {
        Ok(bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect())
    } else if bytes.len() == lanes * 4 {
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f64::from(f32::from_le_bytes(*c)))
            .collect())
    } else {
        Err(GymError::Malformed(format!(
            "reward is {} bytes for {lanes} lanes",
            bytes.len()
        )))
    }
}

/// Maps a lockstep failure onto the runtime's env error, classified as the
/// gRPC env client classifies the same failure.
fn runtime_error(operation: &'static str, error: GymError) -> RuntimeError {
    let recoverable = error.is_recoverable();
    let source = match error {
        GymError::Env(error) => GrpcError::Environment(error.to_wire()),
        GymError::Timeout(t) => GrpcError::Timeout(t),
        GymError::Closed(why) => GrpcError::Transport(TransportError::ConnectFailed(why)),
        other => GrpcError::Protocol(ProtocolError::DecodeError(other.to_string())),
    };
    RuntimeError::env_rpc_with_recoverability(operation, 0, recoverable, source)
}

/// Numbers the runtime's episode ids for the v2 wire.
#[derive(Default)]
struct EpisodeNumbers {
    next: u64,
    by_id: HashMap<String, u64>,
}

impl EpisodeNumbers {
    fn number(&mut self, id: &str) -> u64 {
        if let Some(n) = self.by_id.get(id) {
            return *n;
        }
        self.next += 1;
        self.by_id.insert(id.to_string(), self.next);
        self.next
    }
}

struct Driver {
    link: Arc<dyn LockstepLink>,
    map: GymMap,
    num_envs: usize,
    epoch: AtomicU64,
    token: AtomicU64,
    step: AtomicU64,
    mirror: Mutex<EpisodeMirror>,
    numbers: Mutex<EpisodeNumbers>,
    /// Kept alive for the run (its heartbeats keep the lease), until
    /// [`LockstepEnv::close`] releases it.
    keep: tokio::sync::Mutex<Option<Box<dyn Keep>>>,
}

/// What a driver holds for its run: the session participant whose lease
/// it steps under.
#[async_trait]
pub trait Keep: Send + Sync + 'static {
    /// Releases the lease and detaches (the robot then re-arms for the next
    /// driver).
    async fn release(self: Box<Self>);
}

/// The driver side: a lockstep env as a v1 [`RuntimeEnv`]. Clones share
/// one link (the runtime holds one per lane group).
#[derive(Clone)]
pub struct LockstepEnv {
    inner: Arc<Driver>,
}

impl LockstepEnv {
    /// Drives the env behind `link`, whose robot is `map` (channels bound),
    /// in `epoch` under the lease `token`.
    pub fn new(link: Arc<dyn LockstepLink>, map: GymMap, epoch: Epoch, token: LeaseToken) -> Self {
        Self::with_keep(link, map, epoch, token, None)
    }

    /// [`Self::new`], keeping `keep` (the session participant) alive with
    /// the env.
    pub fn with_keep(
        link: Arc<dyn LockstepLink>,
        map: GymMap,
        epoch: Epoch,
        token: LeaseToken,
        keep: Option<Box<dyn Keep>>,
    ) -> Self {
        let num_envs = (map.info.contract.num_envs as usize).max(1);
        let mirror = EpisodeMirror::new(num_envs, map.info.contract.autoreset_mode);
        Self {
            inner: Arc::new(Driver {
                link,
                map,
                num_envs,
                epoch: AtomicU64::new(epoch.0),
                token: AtomicU64::new(token.0),
                step: AtomicU64::new(0),
                mirror: Mutex::new(mirror),
                numbers: Mutex::new(EpisodeNumbers::default()),
                keep: tokio::sync::Mutex::new(keep),
            }),
        }
    }

    /// The env's contract and handshake.
    pub fn info(&self) -> &GymEnvInfo {
        &self.inner.map.info
    }

    /// The robot map.
    pub fn map(&self) -> &GymMap {
        &self.inner.map
    }

    /// The v1 handshake this env's declaration answers: its contract and
    /// the workflow edition negotiated against `declared` (the runtime's
    /// WANT), exactly as the gRPC env client negotiates it.
    pub fn handshake(&self, declared: Option<&str>) -> Result<EnvHandshake, GymError> {
        let info = &self.inner.map.info;
        let env_offer = rlmesh_proto::SessionOffer {
            editions: info.supported_workflow_editions.clone(),
            preferred: Some(info.workflow_edition.trim().to_string()).filter(|w| !w.is_empty()),
        };
        let runtime_offer = rlmesh_proto::SessionOffer::this_build(declared);
        let selected = rlmesh_proto::negotiate_workflow_edition(&env_offer, &runtime_offer)
            .map_err(|refusal| {
                GymError::Config(format!(
                    "no mutual workflow edition with the env: {refusal}{}",
                    refusal.hint()
                ))
            })?;
        let workflow_edition =
            rlmesh_proto::parse_retained_edition(&selected).map_err(GymError::Config)?;
        let capabilities: Vec<&str> = info.capabilities.iter().map(String::as_str).collect();
        Ok(EnvHandshake {
            env_contract: rlmesh_grpc::wire::env_contract_to_proto(&info.contract),
            num_envs: self.inner.num_envs,
            workflow_edition,
            selected_workflow_edition: selected,
            supported_workflow_editions: env_offer.editions,
            preferred_workflow_edition: env_offer.preferred,
            capabilities: rlmesh_proto::capability_map(&capabilities),
        })
    }

    /// The epoch and lease token its steps carry.
    pub fn authority(&self) -> (Epoch, LeaseToken) {
        (
            Epoch(self.inner.epoch.load(Ordering::Acquire)),
            LeaseToken(self.inner.token.load(Ordering::Acquire)),
        )
    }

    /// Moves to a new epoch and lease (after a re-arm and a fresh grant).
    pub fn set_authority(&self, epoch: Epoch, token: LeaseToken) {
        self.inner.token.store(token.0, Ordering::Release);
        self.inner.epoch.store(epoch.0, Ordering::Release);
    }

    fn lanes(&self, named: &[u32]) -> usize {
        if named.is_empty() {
            self.inner.num_envs
        } else {
            named.len()
        }
    }

    fn timeout(ms: u64) -> Option<Duration> {
        (ms > 0).then(|| Duration::from_millis(ms))
    }

    /// One reset, as the v1 env service answers it.
    pub async fn reset_v1(&self, request: ResetRequest) -> Result<ResetResponse, GymError> {
        let d = &self.inner;
        let lanes = self.lanes(&request.env_indices);
        let episode_ids = {
            let mut numbers = d.numbers.lock().unwrap_or_else(PoisonError::into_inner);
            request
                .episode_ids
                .iter()
                .map(|id| numbers.number(id))
                .collect()
        };
        let reset = LockstepReset {
            epoch: Epoch(d.epoch.load(Ordering::Acquire)),
            lanes: request.env_indices.clone(),
            seeds: request.seeds.iter().map(|s| *s as u64).collect(),
            options_json: meta_json::optional_to_json(
                request.options.clone().map(meta_map_from_proto).as_ref(),
            ),
            episode_ids,
        };
        let reply = d
            .link
            .call(Request::Reset(reset), Self::timeout(request.timeout_ms))
            .await?;
        let observation = map::unpack(
            &d.map.observation,
            &ordered(&d.map.observation, &reply.observations)?,
            lanes,
        )?;
        d.mirror
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .on_reset(&request);
        Ok(ResetResponse {
            observation: Some(SpaceValue {
                leaves: observation,
            }),
            infos: meta_json::optional_from_json(&reply.info_json)
                .map_err(GymError::Malformed)?
                .as_ref()
                .map(meta_map_to_proto),
        })
    }

    /// One step, as the v1 env service answers it.
    pub async fn step_v1(&self, request: StepRequest) -> Result<StepResponse, GymError> {
        let d = &self.inner;
        let lanes = self.lanes(&request.env_indices);
        let action = request
            .action
            .as_ref()
            .map(|a| a.leaves.as_slice())
            .unwrap_or_default();
        let commands = tagged(&d.map.action, map::pack(&d.map.action, action, lanes)?);
        let step = LockstepStep {
            epoch: Epoch(d.epoch.load(Ordering::Acquire)),
            step: d.step.fetch_add(1, Ordering::Relaxed) + 1,
            lease_token: LeaseToken(d.token.load(Ordering::Acquire)),
            lanes: request.env_indices.clone(),
            commands,
        };
        let reply = d
            .link
            .call(Request::Step(step), Self::timeout(request.timeout_ms))
            .await?;
        let observation = map::unpack(
            &d.map.observation,
            &ordered(&d.map.observation, &reply.observations)?,
            lanes,
        )?;
        let mut response = StepResponse {
            observation: Some(SpaceValue {
                leaves: observation,
            }),
            rewards: rewards_from(&reply.reward, lanes)?,
            terminated_mask: reply.terminated.to_vec(),
            truncated_mask: reply.truncated.to_vec(),
            infos: meta_json::optional_from_json(&reply.info_json)
                .map_err(GymError::Malformed)?
                .as_ref()
                .map(meta_map_to_proto),
            completed_episodes: Vec::new(),
            env_indices: request.env_indices.clone(),
        };
        d.mirror
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .on_step(&request, &mut response);
        Ok(response)
    }

    /// One render frame of lane `request.env_indices[0]`, as the v1 env
    /// service answers it.
    pub async fn render_v1(&self, request: RenderRequest) -> Result<RenderResponse, GymError> {
        let timeout = Self::timeout(request.timeout_ms);
        let lane = request.env_indices.first().copied().unwrap_or(0);
        let frame = self.inner.link.render(lane, timeout).await?;
        Ok(rlmesh_grpc::wire::render_result_to_proto(
            &rlmesh_spaces::RenderResult {
                frame: frame.map(|frame| rlmesh_spaces::RenderFrame {
                    frame: frame.to_vec(),
                }),
            },
        ))
    }

    /// Releases the lease and detaches (idempotent); every clone is closed.
    pub async fn close(&self) {
        if let Some(keep) = self.inner.keep.lock().await.take() {
            keep.release().await;
        }
    }

    /// The channel of every commandable part.
    pub fn command_channels(&self) -> Vec<ChannelId> {
        self.inner
            .map
            .action
            .parts
            .iter()
            .map(|p| p.spec.channel)
            .collect()
    }
}

#[async_trait]
impl RuntimeEnv for LockstepEnv {
    async fn reset(&mut self, request: ResetRequest) -> Result<RuntimeEnvReset, RuntimeError> {
        let response = self
            .reset_v1(request)
            .await
            .map_err(|e| runtime_error("env.reset", e))?;
        Ok(RuntimeEnvReset {
            response,
            endpoint_total_ns: None,
            phases: Default::default(),
        })
    }

    async fn step(&mut self, request: StepRequest) -> Result<RuntimeEnvStep, RuntimeError> {
        let response = self
            .step_v1(request)
            .await
            .map_err(|e| runtime_error("env.step", e))?;
        Ok(RuntimeEnvStep {
            response,
            endpoint_total_ns: None,
            phases: Default::default(),
        })
    }

    async fn close(&mut self, timeout: Duration) -> Result<(), String> {
        tokio::time::timeout(timeout, LockstepEnv::close(self))
            .await
            .map_err(|_| format!("the lockstep driver did not detach within {timeout:?}"))
    }
}
