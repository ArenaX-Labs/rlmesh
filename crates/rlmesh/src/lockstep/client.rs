//! The driver side of the Gym path (§12.4): [`LockstepClient`] resets,
//! steps and renders a lockstep Gym env on native values, rebuilding its
//! exact spaces from the robot's `gym_env`.
//!
//! The client numbers the episodes its resets start (1, 2, ...) for
//! `LockstepReset.episode_ids`; a reply carries no episode metadata, so
//! accounting (ids, completions) is the caller's.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use rlmesh_session::wire::{
    ChannelId, EndpointTiming, Epoch, LeaseToken, LockstepReset, LockstepStep,
};
use rlmesh_spaces::leaves::{decode_leaf_slab, encode_leaf_slab};
use rlmesh_spaces::{MetaMap, SpaceSpec, SpaceValue};

use crate::lockstep::error::GymError;
use crate::lockstep::exchange::{LockstepLink, Request};
use crate::lockstep::map::{self, GymEnvInfo, GymMap, Side};
use crate::lockstep::meta_json;
use crate::lockstep::served::{Reset, Step, ordered, tagged};

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

fn flags(bytes: &[u8]) -> Vec<bool> {
    bytes.iter().map(|b| *b != 0).collect()
}

/// A step's answer with the observation as leaf slabs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LeafStep {
    /// The lanes' next observation, as leaf slabs.
    pub observation: Vec<Bytes>,
    /// One reward per lane.
    pub rewards: Vec<f64>,
    /// Per-lane `terminated`.
    pub terminated: Vec<bool>,
    /// Per-lane `truncated`.
    pub truncated: Vec<bool>,
    /// The info.
    pub info: Option<MetaMap>,
}

/// What a driver holds for its run: the session participant whose lease
/// it steps under.
#[async_trait]
pub trait Keep: Send + Sync + 'static {
    /// Releases the lease and detaches (the robot then re-arms for the next
    /// driver).
    async fn release(self: Box<Self>);
}

struct Driver {
    link: Arc<dyn LockstepLink>,
    map: GymMap,
    num_envs: usize,
    epoch: AtomicU64,
    token: AtomicU64,
    step: AtomicU64,
    episodes: AtomicU64,
    /// Kept alive for the run (its heartbeats keep the lease), until
    /// [`LockstepClient::close`] releases it.
    keep: tokio::sync::Mutex<Option<Box<dyn Keep>>>,
    /// What the env measured answering the last reset or step.
    last: std::sync::Mutex<Option<EndpointTiming>>,
}

/// A driver of a lockstep Gym env. Clones share one link.
#[derive(Clone)]
pub struct LockstepClient {
    inner: Arc<Driver>,
}

impl LockstepClient {
    /// Drives the env behind `link`, whose robot is `map` (channels bound),
    /// in `epoch` under the lease `token`.
    pub fn new(link: Arc<dyn LockstepLink>, map: GymMap, epoch: Epoch, token: LeaseToken) -> Self {
        Self::with_keep(link, map, epoch, token, None)
    }

    /// [`Self::new`], keeping `keep` (the session participant) alive with
    /// the client.
    pub fn with_keep(
        link: Arc<dyn LockstepLink>,
        map: GymMap,
        epoch: Epoch,
        token: LeaseToken,
        keep: Option<Box<dyn Keep>>,
    ) -> Self {
        let num_envs = (map.info.contract.num_envs as usize).max(1);
        Self {
            inner: Arc::new(Driver {
                link,
                map,
                num_envs,
                epoch: AtomicU64::new(epoch.0),
                token: AtomicU64::new(token.0),
                step: AtomicU64::new(0),
                episodes: AtomicU64::new(0),
                keep: tokio::sync::Mutex::new(keep),
                last: std::sync::Mutex::default(),
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

    /// What the env measured answering the last reset or step (v1's
    /// endpoint phases), when it measured.
    pub fn last_timing(&self) -> Option<EndpointTiming> {
        *self
            .inner
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn timed(&self, timing: Option<EndpointTiming>) {
        *self
            .inner
            .last
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = timing;
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

    fn lanes(&self, named: &[u32]) -> usize {
        if named.is_empty() {
            self.inner.num_envs
        } else {
            named.len()
        }
    }

    fn space(spec: &Option<SpaceSpec>) -> Result<&SpaceSpec, GymError> {
        spec.as_ref()
            .ok_or_else(|| GymError::Map("the env has no space".into()))
    }

    /// The observation leaf slabs of `lanes` lanes in `side`'s part slabs.
    fn observation_leaves(
        &self,
        values: &[rlmesh_session::wire::PartValues],
        lanes: usize,
    ) -> Result<Vec<Bytes>, GymError> {
        let side: &Side = &self.inner.map.observation;
        map::unpack(side, &ordered(side, values)?, lanes)
    }

    fn decode(&self, leaves: &[Bytes], lanes: usize) -> Result<Vec<SpaceValue>, GymError> {
        let space = Self::space(&self.inner.map.info.contract.observation_space)?;
        decode_leaf_slab(leaves, space, lanes).map_err(|e| GymError::Malformed(e.to_string()))
    }

    /// Resets `lanes` (empty: every lane), `seeds` positional to them.
    pub async fn reset(
        &self,
        lanes: &[u32],
        seeds: &[i64],
        options: Option<&MetaMap>,
        timeout: Option<Duration>,
    ) -> Result<Reset, GymError> {
        let width = self.lanes(lanes);
        let (leaves, info) = self.reset_leaves(lanes, seeds, options, timeout).await?;
        Ok(Reset {
            observations: self.decode(&leaves, width)?,
            info,
        })
    }

    /// [`Self::reset`], the observation as leaf slabs.
    pub async fn reset_leaves(
        &self,
        lanes: &[u32],
        seeds: &[i64],
        options: Option<&MetaMap>,
        timeout: Option<Duration>,
    ) -> Result<(Vec<Bytes>, Option<MetaMap>), GymError> {
        let d = &self.inner;
        let width = self.lanes(lanes);
        let first = d.episodes.fetch_add(width as u64, Ordering::Relaxed) + 1;
        let reset = LockstepReset {
            epoch: Epoch(d.epoch.load(Ordering::Acquire)),
            lanes: lanes.to_vec(),
            // i64 for the env, u64 on the v2 wire: the same 64 bits.
            seeds: seeds.iter().map(|s| *s as u64).collect(),
            options_json: meta_json::optional_to_json(options),
            episode_ids: (first..first + width as u64).collect(),
        };
        let reply = d.link.call(Request::Reset(reset), timeout).await?;
        self.timed(reply.timing);
        Ok((
            self.observation_leaves(&reply.observations, width)?,
            meta_json::optional_from_json(&reply.info_json).map_err(GymError::Malformed)?,
        ))
    }

    /// Steps `lanes` (empty: every lane) with one action each.
    pub async fn step(
        &self,
        lanes: &[u32],
        actions: &[SpaceValue],
        timeout: Option<Duration>,
    ) -> Result<Step, GymError> {
        let space = Self::space(&self.inner.map.info.contract.action_space)?;
        let leaves =
            encode_leaf_slab(actions, space).map_err(|e| GymError::Value(e.to_string()))?;
        let step = self.step_leaves(lanes, &leaves, timeout).await?;
        Ok(Step {
            observations: self.decode(&step.observation, self.lanes(lanes))?,
            rewards: step.rewards,
            terminated: step.terminated,
            truncated: step.truncated,
            info: step.info,
        })
    }

    /// [`Self::step`] on leaf slabs: one action per lane in, the next
    /// observation out.
    pub async fn step_leaves(
        &self,
        lanes: &[u32],
        leaves: &[Bytes],
        timeout: Option<Duration>,
    ) -> Result<LeafStep, GymError> {
        let d = &self.inner;
        let width = self.lanes(lanes);
        let commands = tagged(&d.map.action, map::pack(&d.map.action, leaves, width)?);
        let step = LockstepStep {
            epoch: Epoch(d.epoch.load(Ordering::Acquire)),
            step: d.step.fetch_add(1, Ordering::Relaxed) + 1,
            lease_token: LeaseToken(d.token.load(Ordering::Acquire)),
            lanes: lanes.to_vec(),
            commands,
        };
        let reply = d.link.call(Request::Step(step), timeout).await?;
        self.timed(reply.timing);
        Ok(LeafStep {
            observation: self.observation_leaves(&reply.observations, width)?,
            rewards: rewards_from(&reply.reward, width)?,
            terminated: flags(&reply.terminated),
            truncated: flags(&reply.truncated),
            info: meta_json::optional_from_json(&reply.info_json).map_err(GymError::Malformed)?,
        })
    }

    /// Lane `lane`'s frame as PNG bytes, if it has one.
    pub async fn render(
        &self,
        lane: u32,
        timeout: Option<Duration>,
    ) -> Result<Option<Bytes>, GymError> {
        self.inner.link.render(lane, timeout).await
    }

    /// Releases the lease and detaches (idempotent); every clone is closed.
    pub async fn close(&self) {
        if let Some(keep) = self.inner.keep.lock().await.take() {
            keep.release().await;
        }
    }
}
