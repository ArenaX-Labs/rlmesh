//! The env side of the Gym path: a served env behind the lockstep
//! exchange, on native values.
//!
//! - **Reward width.** Gym rewards are `f64` and the Gym path sends `f64 [N]` so they stay
//!   bit-exact; readers take `f32 [N]` or `f64 [N]` by length.
//! - **Info.** `info_json` always carries the env's `info` ([`mod@crate::lockstep::meta_json`]);
//!   empty means unset.
//! - **Episode ids.** A driver numbers its episodes (1, 2, ...) in `LockstepReset.episode_ids`; a
//!   `LockstepStep` has no id field and a reply's `episode_ids` stays empty.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_session::wire::timing::ns;
use rlmesh_session::wire::{
    EndpointTiming, LockstepReply, LockstepReset, LockstepStep, PartValues,
};
use rlmesh_spaces::leaves::{decode_leaf_slab, encode_leaf_slab};
use rlmesh_spaces::{EnvContract, MetaMap, SpaceSpec, SpaceValue};

use crate::lockstep::error::GymError;
use crate::lockstep::exchange::{LockstepHandler, Request};
use crate::lockstep::map::{self, GymMap, Side};
use crate::lockstep::meta_json;
use crate::lockstep::record::Tee;
use crate::{EnvironmentError, ErrorCode};

/// A reset's answer: one observation per lane reset, and the info.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reset {
    /// One observation per lane.
    pub observations: Vec<SpaceValue>,
    /// The info.
    pub info: Option<MetaMap>,
}

/// A step's answer, one entry per lane stepped.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    /// One observation per lane.
    pub observations: Vec<SpaceValue>,
    /// One reward per lane.
    pub rewards: Vec<f64>,
    /// Per-lane `terminated`.
    pub terminated: Vec<bool>,
    /// Per-lane `truncated`.
    pub truncated: Vec<bool>,
    /// The info.
    pub info: Option<MetaMap>,
}

/// An env as the lockstep path serves it. `lanes` names lanes by index
/// (empty: every lane); a reset's `seeds` are positional to them.
#[async_trait]
pub trait Served: Send + Sync + 'static {
    /// The contract, its `num_envs` the env's lane count.
    fn contract(&self) -> EnvContract;

    /// Whether lanes reset and step individually (v1's `subset_step`).
    fn subset_step(&self) -> bool;

    /// Resets `lanes`.
    async fn reset(
        &self,
        lanes: &[u32],
        seeds: Vec<i64>,
        options: Option<MetaMap>,
    ) -> Result<Reset, EnvironmentError>;

    /// Steps `lanes` with one action each.
    async fn step(&self, lanes: &[u32], actions: Vec<SpaceValue>)
    -> Result<Step, EnvironmentError>;

    /// Lane `lane`'s frame as PNG bytes, if it has one.
    async fn render(&self, lane: u32) -> Result<Option<Vec<u8>>, EnvironmentError>;

    /// Closes the env.
    async fn close(&self) -> Result<(), EnvironmentError>;
}

pub(crate) fn tagged(side: &Side, slabs: Vec<Bytes>) -> Vec<PartValues> {
    side.parts
        .iter()
        .zip(slabs)
        .map(|(part, payload)| PartValues {
            channel: part.spec.channel,
            payload,
        })
        .collect()
}

pub(crate) fn ordered(side: &Side, values: &[PartValues]) -> Result<Vec<Bytes>, GymError> {
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

fn rewards_bytes(rewards: &[f64], lanes: usize) -> Bytes {
    let mut out = Vec::with_capacity(lanes * 8);
    for lane in 0..lanes {
        out.extend_from_slice(&rewards.get(lane).copied().unwrap_or(0.0).to_le_bytes());
    }
    Bytes::from(out)
}

fn mask(flags: &[bool], lanes: usize) -> Bytes {
    let mut out = vec![0_u8; lanes];
    for (lane, flag) in flags.iter().take(lanes).enumerate() {
        out[lane] = u8::from(*flag);
    }
    Bytes::from(out)
}

fn refused(code: ErrorCode, message: String) -> GymError {
    GymError::env(code, message, true)
}

/// The env side: a served env answering lockstep requests.
pub struct GymEnv {
    env: Arc<dyn Served>,
    map: GymMap,
    num_envs: usize,
    tee: Option<Tee>,
}

impl GymEnv {
    /// Serves `env` with `map` (channels bound).
    pub fn new(env: Arc<dyn Served>, map: GymMap) -> Self {
        let num_envs = (map.info.contract.num_envs as usize).max(1);
        Self {
            env,
            map,
            num_envs,
            tee: None,
        }
    }

    /// Tees every request and reply into a recording.
    #[must_use]
    pub fn with_tee(mut self, tee: Tee) -> Self {
        self.tee = Some(tee);
        self
    }

    /// The map it serves.
    pub fn map(&self) -> &GymMap {
        &self.map
    }

    fn space(spec: &Option<SpaceSpec>) -> Result<&SpaceSpec, GymError> {
        spec.as_ref()
            .ok_or_else(|| GymError::Map("the env has no space".into()))
    }

    fn width(&self, op: &str, lanes: &[u32], seeds: usize) -> Result<usize, GymError> {
        if lanes.is_empty() {
            return Ok(self.num_envs);
        }
        if !self.env.subset_step() {
            return Err(refused(
                ErrorCode::Unsupported,
                match op {
                    "partial reset" => "partial reset (ResetRequest.env_indices) is not supported \
                                        by this environment"
                        .into(),
                    _ => "subset stepping (StepRequest.env_indices) is not supported by this \
                          environment"
                        .into(),
                },
            ));
        }
        if seeds != 0 && seeds != lanes.len() {
            return Err(refused(
                ErrorCode::InvalidAction,
                format!(
                    "{op}: seeds length {seeds} does not match env_indices length {} (provide \
                     one seed per reset lane, or none)",
                    lanes.len()
                ),
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(lanes.len());
        for &lane in lanes {
            if lane as usize >= self.num_envs {
                return Err(refused(
                    ErrorCode::InvalidAction,
                    format!(
                        "{op}: env_index {lane} out of range for num_envs {}",
                        self.num_envs
                    ),
                ));
            }
            if !seen.insert(lane) {
                return Err(refused(
                    ErrorCode::InvalidAction,
                    format!("{op}: duplicate env_index {lane}"),
                ));
            }
        }
        Ok(lanes.len())
    }

    fn observations(&self, values: &[SpaceValue], lanes: usize) -> Result<Vec<Bytes>, GymError> {
        let space = Self::space(&self.map.info.contract.observation_space)?;
        let leaves = encode_leaf_slab(values, space)
            .map_err(|e| GymError::env(ErrorCode::Internal, e.to_string(), false))?;
        map::pack(&self.map.observation, &leaves, lanes)
    }

    async fn reset(&self, reset: LockstepReset) -> Result<LockstepReply, GymError> {
        let lanes = self.width("partial reset", &reset.lanes, reset.seeds.len())?;
        let options =
            meta_json::optional_from_json(&reset.options_json).map_err(GymError::Malformed)?;
        let seeds = reset.seeds.iter().map(|s| *s as i64).collect();
        let called = std::time::Instant::now();
        let ok = self
            .env
            .reset(&reset.lanes, seeds, options)
            .await
            .map_err(GymError::Env)?;
        let user_ns = ns(called.elapsed());
        let encoded = std::time::Instant::now();
        let slabs = self.observations(&ok.observations, lanes)?;
        Ok(LockstepReply {
            timing: Some(EndpointTiming {
                user_ns,
                encode_ns: ns(encoded.elapsed()),
                ..EndpointTiming::default()
            }),
            epoch: reset.epoch,
            step: 0,
            observations: tagged(&self.map.observation, slabs),
            reward: Bytes::new(),
            terminated: Bytes::new(),
            truncated: Bytes::new(),
            episode_ids: reset.episode_ids,
            info_json: meta_json::optional_to_json(ok.info.as_ref()),
        })
    }

    async fn step(&self, step: &LockstepStep) -> Result<LockstepReply, GymError> {
        let lanes = self.width("subset step", &step.lanes, 0)?;
        let decoded = std::time::Instant::now();
        let leaves = map::unpack(
            &self.map.action,
            &ordered(&self.map.action, &step.commands)?,
            lanes,
        )?;
        let space = Self::space(&self.map.info.contract.action_space)?;
        let actions = decode_leaf_slab(&leaves, space, lanes)
            .map_err(|e| GymError::env(ErrorCode::InvalidAction, e.to_string(), true))?;
        let decode_ns = ns(decoded.elapsed());
        let called = std::time::Instant::now();
        let ok = self
            .env
            .step(&step.lanes, actions)
            .await
            .map_err(GymError::Env)?;
        let user_ns = ns(called.elapsed());
        let encoded = std::time::Instant::now();
        let slabs = self.observations(&ok.observations, lanes)?;
        Ok(LockstepReply {
            timing: Some(EndpointTiming {
                decode_ns,
                user_ns,
                encode_ns: ns(encoded.elapsed()),
                ..EndpointTiming::default()
            }),
            epoch: step.epoch,
            step: step.step,
            observations: tagged(&self.map.observation, slabs),
            reward: rewards_bytes(&ok.rewards, lanes),
            terminated: mask(&ok.terminated, lanes),
            truncated: mask(&ok.truncated, lanes),
            episode_ids: Vec::new(),
            info_json: meta_json::optional_to_json(ok.info.as_ref()),
        })
    }
}

#[async_trait]
impl LockstepHandler for GymEnv {
    async fn flush(&self) {
        if let Some(tee) = &self.tee {
            tee.flush().await;
        }
    }

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

    async fn render(&self, lane: u32) -> Result<Option<Bytes>, GymError> {
        let frame = self.env.render(lane).await.map_err(GymError::Env)?;
        Ok(frame.map(Bytes::from))
    }

    fn concurrent(&self) -> bool {
        self.env.subset_step()
    }
}
