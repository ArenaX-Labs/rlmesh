//! The Gym compatibility path (§12.4): a v1 env served, and a v1 runtime
//! driving it, over the lockstep exchange.
//!
//! - [`GymEnv`] is the env side: a v1 [`Environment`] behind the v1 env
//!   service's own core ([`EnvCore`]), so a `LockstepStep` runs exactly what
//!   the same Join step runs (validation, the NEXT_STEP lane contract,
//!   partial resets, episode accounting). Leaf slabs become part slabs and
//!   back without conversion (bit-exact).
//! - [`LockstepEnv`] is the driver side: a [`RuntimeEnv`], so the v1
//!   [`RuntimeDriver`](rlmesh_runtime::RuntimeDriver) drives a lockstep env
//!   with its chunking, prefetch, batching, autoreset handling and
//!   driver-minted episode ids unchanged. The lockstep reply carries no
//!   episode metadata; an [`EpisodeMirror`] rebuilds the completions the
//!   env server would have reported.
//!
//! Three readings of §12.2 the compatibility path needs:
//!
//! - **Reward width.** v1 rewards are `f64`; the compatibility path sends
//!   `f64 [N]` so they stay bit-exact. Readers take `f32 [N]` or `f64 [N]`
//!   by length.
//! - **Info.** The v1 runtime reads `info` (success, final info), so the
//!   compatibility path always records it: `info_json` carries the v1
//!   `MetaMap` ([`meta_json`](mod@crate::lockstep::meta_json)); empty means unset.
//! - **Episode ids.** v1 mints UUIDv7 strings and pushes a fresh one down
//!   on the step that rolls a NEXT_STEP autoreset; v2 ids are `u64` and a
//!   `LockstepStep` has no id field. The driver numbers its ids (1, 2, ...)
//!   for `LockstepReset.episode_ids` and keeps the v1 strings itself; the
//!   env sees the numbers as text, and a reply's `episode_ids` stays empty.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_grpc::env::{
    EnvCore, EnvHandshake, Environment, EpisodeMirror, proto_error_to_env_error,
};
use rlmesh_grpc::error::{Error as GrpcError, ProtocolError, TransportError};
use rlmesh_proto::env::v1::{
    JoinRequest, RenderRequest, RenderResponse, ResetRequest, ResetResponse, StepRequest,
    StepResponse, join_request, join_response,
};
use rlmesh_proto::spaces::v1::SpaceValue;
use rlmesh_runtime::{RuntimeEnv, RuntimeEnvReset, RuntimeEnvStep, RuntimeError};
use rlmesh_session::wire::{
    ChannelId, Epoch, LeaseToken, LockstepReply, LockstepReset, LockstepStep, PartValues,
};

use crate::lockstep::error::GymError;
use crate::lockstep::exchange::{LockstepHandler, LockstepLink, Request};
use crate::lockstep::map::{self, GymEnvInfo, GymMap, Side};
use crate::lockstep::meta_json;
use crate::lockstep::record::Tee;

/// What a v1 env tells a handshake, for its `gym_env`.
pub fn env_info<E: Environment>(env: &E, declared_edition: Option<&str>) -> GymEnvInfo {
    let mut contract = env.env_contract().clone();
    contract.num_envs = u32::try_from(env.num_envs()).unwrap_or(u32::MAX);
    GymEnvInfo {
        contract,
        workflow_edition: rlmesh_proto::declared_workflow_edition(declared_edition).to_string(),
        supported_workflow_editions: rlmesh_proto::supported_workflow_editions(),
        capabilities: if env.supports_lanes() {
            vec![rlmesh_proto::capabilities::ENV_SUBSET_STEP.to_string()]
        } else {
            Vec::new()
        },
    }
}

/// The slabs of a side's parts, by channel.
fn tagged(side: &Side, slabs: Vec<Bytes>) -> Vec<PartValues> {
    side.parts
        .iter()
        .zip(slabs)
        .map(|(part, payload)| PartValues {
            channel: part.spec.channel,
            payload,
        })
        .collect()
}

/// The slabs of a side's parts, in part order, from messages by channel.
fn ordered(side: &Side, values: &[PartValues]) -> Result<Vec<Bytes>, GymError> {
    side.parts
        .iter()
        .map(|part| {
            values
                .iter()
                .find(|v| v.channel == part.spec.channel)
                .map(|v| v.payload.clone())
                .ok_or_else(|| GymError::Malformed(format!("no slab for part {}", part.spec.path)))
        })
        .collect()
}

fn leaves(value: Option<SpaceValue>) -> Vec<Bytes> {
    value.map(|v| v.leaves).unwrap_or_default()
}

fn rewards_bytes(rewards: &[f64], lanes: usize) -> Bytes {
    let mut out = Vec::with_capacity(lanes * 8);
    for lane in 0..lanes {
        out.extend_from_slice(&rewards.get(lane).copied().unwrap_or(0.0).to_le_bytes());
    }
    Bytes::from(out)
}

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

fn mask(bytes: &[u8], lanes: usize) -> Bytes {
    let mut out = vec![0_u8; lanes];
    for (lane, b) in bytes.iter().take(lanes).enumerate() {
        out[lane] = u8::from(*b != 0);
    }
    Bytes::from(out)
}

/// The env side: a v1 env answering lockstep requests.
pub struct GymEnv<E: Environment> {
    core: EnvCore<E>,
    map: GymMap,
    num_envs: usize,
    request_id: AtomicU64,
    tee: Option<Tee>,
}

impl<E: Environment + 'static> GymEnv<E> {
    /// Serves `env` with `map` (channels bound).
    pub fn new(env: Arc<E>, map: GymMap) -> Self {
        let num_envs = env.num_envs().max(1);
        Self {
            core: EnvCore::new(env),
            map,
            num_envs,
            request_id: AtomicU64::new(1),
            tee: None,
        }
    }

    /// Tees every request and reply into a recording (§13).
    #[must_use]
    pub fn with_tee(mut self, tee: Tee) -> Self {
        self.tee = Some(tee);
        self
    }

    /// The map it serves.
    pub fn map(&self) -> &GymMap {
        &self.map
    }

    /// Completes every open episode (the session ended).
    pub async fn complete_all(&self, reason: &str) {
        let _ = self.core.complete_all(reason).await;
    }

    async fn call(&self, kind: join_request::Kind) -> Result<join_response::Kind, GymError> {
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed).to_string();
        let response = self
            .core
            .handle(JoinRequest {
                request_id,
                kind: Some(kind),
            })
            .await;
        match response.kind {
            Some(join_response::Kind::Error(error)) => Err(GymError::Env(error)),
            Some(kind) => Ok(kind),
            None => Err(GymError::Malformed("an empty env response".into())),
        }
    }

    async fn reset(&self, reset: LockstepReset) -> Result<LockstepReply, GymError> {
        let lanes = if reset.lanes.is_empty() {
            self.num_envs
        } else {
            reset.lanes.len()
        };
        let request = ResetRequest {
            // u64 on the v2 wire, i64 in v1: the same 64 bits.
            seeds: reset.seeds.iter().map(|s| *s as i64).collect(),
            options: meta_json::optional_from_json(&reset.options_json)
                .map_err(GymError::Malformed)?,
            timeout_ms: 0,
            env_indices: reset.lanes.clone(),
            episode_ids: reset.episode_ids.iter().map(u64::to_string).collect(),
        };
        let join_response::Kind::Reset(ok) = self.call(join_request::Kind::Reset(request)).await?
        else {
            return Err(GymError::Malformed("not a reset response".into()));
        };
        let slabs = map::pack(&self.map.observation, &leaves(ok.observation), lanes)?;
        Ok(LockstepReply {
            epoch: reset.epoch,
            step: 0,
            observations: tagged(&self.map.observation, slabs),
            reward: Bytes::new(),
            terminated: Bytes::new(),
            truncated: Bytes::new(),
            episode_ids: reset.episode_ids,
            info_json: meta_json::optional_to_json(ok.infos.as_ref()),
        })
    }

    async fn step(&self, step: &LockstepStep) -> Result<LockstepReply, GymError> {
        let lanes = if step.lanes.is_empty() {
            self.num_envs
        } else {
            step.lanes.len()
        };
        let action = map::unpack(
            &self.map.action,
            &ordered(&self.map.action, &step.commands)?,
            lanes,
        )?;
        let request = StepRequest {
            action: Some(SpaceValue { leaves: action }),
            timeout_ms: 0,
            env_indices: step.lanes.clone(),
            episode_ids: Vec::new(),
        };
        let join_response::Kind::Step(ok) = self.call(join_request::Kind::Step(request)).await?
        else {
            return Err(GymError::Malformed("not a step response".into()));
        };
        let slabs = map::pack(&self.map.observation, &leaves(ok.observation), lanes)?;
        Ok(LockstepReply {
            epoch: step.epoch,
            step: step.step,
            observations: tagged(&self.map.observation, slabs),
            reward: rewards_bytes(&ok.rewards, lanes),
            terminated: mask(&ok.terminated_mask, lanes),
            truncated: mask(&ok.truncated_mask, lanes),
            episode_ids: Vec::new(),
            info_json: meta_json::optional_to_json(ok.infos.as_ref()),
        })
    }
}

#[async_trait]
impl<E: Environment + 'static> LockstepHandler for GymEnv<E> {
    async fn handle(&self, request: Request) -> Result<LockstepReply, GymError> {
        let tee = self.tee.as_ref().map(|_| request.clone());
        let reply = match request {
            Request::Reset(reset) => self.reset(reset).await,
            Request::Step(step) => self.step(&step).await,
        };
        if let (Some(tee), Some(request), Ok(reply)) = (&self.tee, tee, &reply) {
            tee.record(request, reply.clone());
        }
        reply
    }

    async fn render(&self, request: RenderRequest) -> Result<RenderResponse, GymError> {
        let join_response::Kind::Render(frame) =
            self.call(join_request::Kind::Render(request)).await?
        else {
            return Err(GymError::Malformed("not a render response".into()));
        };
        Ok(frame)
    }

    fn concurrent(&self) -> bool {
        self.core.env().supports_lanes()
    }
}

/// Maps a lockstep failure onto the runtime's env error, classified as the
/// gRPC env client classifies the same failure.
fn runtime_error(operation: &'static str, error: GymError) -> RuntimeError {
    let recoverable = error.is_recoverable();
    let source = match error {
        GymError::Env(error) => GrpcError::Environment(proto_error_to_env_error(error)),
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
            options_json: meta_json::optional_to_json(request.options.as_ref()),
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
            infos: meta_json::optional_from_json(&reply.info_json).map_err(GymError::Malformed)?,
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
            infos: meta_json::optional_from_json(&reply.info_json).map_err(GymError::Malformed)?,
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
        self.inner.link.render(request, timeout).await
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
