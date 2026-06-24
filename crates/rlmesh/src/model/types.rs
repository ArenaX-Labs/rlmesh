use std::sync::Arc;

use rlmesh_grpc::wire::Bytes;

use crate::spaces;
use crate::{Error, Result};

/// One sub-environment's position within a routed predict request.
///
/// A vectorized predict carries one slot per active sub-environment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteSlot {
    /// Identifier of the episode this slot belongs to.
    pub episode_id: String,
    /// Index of the sub-environment.
    pub env_index: i32,
    /// Step number within the episode.
    pub step: i64,
    /// Whether this slot is the first step of a new episode.
    pub reset: bool,
}

/// Routing metadata attached to a [`ModelObservation`].
///
/// Identifies the session/route the request belongs to and the per-slot episode
/// state. The `primary_*` accessors read the first slot, which is the natural
/// choice for single-env handlers.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteContext {
    /// Identifier of the session this request belongs to.
    pub session_id: String,
    /// Identifier of the route within the session.
    pub route_id: String,
    /// Identifier of this individual request.
    pub request_id: String,
    /// One slot per active sub-environment.
    pub slots: Vec<ModelRouteSlot>,
}

impl ModelRouteContext {
    /// The primary (first) slot's episode id, or `""` if there are no slots.
    pub fn primary_episode_id(&self) -> &str {
        self.primary_slot()
            .map(|slot| slot.episode_id.as_str())
            .unwrap_or("")
    }

    /// The primary slot's env index, or `0` if there are no slots.
    pub fn primary_env_index(&self) -> i32 {
        self.primary_slot().map_or(0, |slot| slot.env_index)
    }

    /// The primary slot's step number, or `0` if there are no slots.
    pub fn primary_step(&self) -> i64 {
        self.primary_slot().map_or(0, |slot| slot.step)
    }

    /// Whether the primary slot marks the start of a new episode.
    pub fn primary_reset(&self) -> bool {
        self.primary_slot().is_some_and(|slot| slot.reset)
    }

    /// The episode id of every slot, in slot order.
    pub fn episode_ids(&self) -> Vec<String> {
        self.slots
            .iter()
            .map(|slot| slot.episode_id.clone())
            .collect()
    }

    /// The primary (first) slot, if any.
    pub fn primary_slot(&self) -> Option<&ModelRouteSlot> {
        self.slots.first()
    }
}

/// An observation delivered to [`ModelHandler::predict`](crate::ModelHandler::predict).
///
/// Carries the encoded observation payload plus the routing context and the
/// env contract / batch size needed to decode it. The convenience accessors
/// forward to the primary slot of [`route`](ModelObservation::route).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelObservation {
    /// Raw per-leaf observation wire bytes, if present. `None` is the only
    /// "absent value" signal (§5) — decode via
    /// [`decoded`](ModelObservation::decoded)/[`decoded_lanes`](ModelObservation::decoded_lanes),
    /// never read these bytes directly.
    pub observation: Option<Vec<Bytes>>,
    /// Routing/episode metadata for this request.
    pub route: ModelRouteContext,
    /// Whether this request starts a new episode.
    pub reset: bool,
    /// Number of sub-environments in this route's batch.
    pub num_envs: usize,
    /// The env contract (spaces/metadata) for decoding the observation. Shared
    /// (`Arc`) so the per-predict hot path clones a refcount, not the contract.
    pub env_contract: Option<Arc<spaces::EnvContract>>,
}

impl ModelObservation {
    /// The primary slot's episode id (see [`ModelRouteContext::primary_episode_id`]).
    pub fn episode_id(&self) -> &str {
        self.route.primary_episode_id()
    }

    /// The episode id of every slot (see [`ModelRouteContext::episode_ids`]).
    pub fn episode_ids(&self) -> Vec<String> {
        self.route.episode_ids()
    }

    /// The primary slot's step number.
    pub fn step(&self) -> i64 {
        self.route.primary_step()
    }

    /// The primary slot's env index.
    pub fn env_index(&self) -> i32 {
        self.route.primary_env_index()
    }

    /// Decode the observation into one typed [`SpaceValue`](spaces::SpaceValue) per lane (length
    /// `== num_envs`), using the route's pinned observation space.
    ///
    /// Errors if the observation is absent — an absent value yields no lanes, so
    /// returning it as an empty `Vec` would silently violate the `== num_envs`
    /// contract. Guard with `observation.is_some()` for optional observations.
    pub fn decoded_lanes(&self) -> Result<Vec<spaces::SpaceValue>> {
        let leaves = self.observation.as_ref().ok_or_else(|| {
            Error::model("observation absent; cannot decode lanes (check is_some() first)")
        })?;
        let contract = self
            .env_contract
            .as_ref()
            .ok_or_else(|| Error::model("observation missing env contract; cannot decode"))?;
        let space = contract
            .observation_space
            .as_ref()
            .ok_or_else(|| Error::model("env contract missing observation space"))?;
        let value = rlmesh_grpc::wire::leaves_value(leaves.clone());
        rlmesh_grpc::wire::decode_batched_partial_values(Some(&value), space, self.num_envs)
            .map_err(|err| Error::model(err.to_string()))
    }

    /// Decode a single-env observation. Errors unless exactly one lane is present.
    pub fn decoded(&self) -> Result<spaces::SpaceValue> {
        let mut lanes = self.decoded_lanes()?;
        if lanes.len() != 1 {
            return Err(Error::model(format!(
                "decoded() requires a single-env observation, got {} lanes",
                lanes.len()
            )));
        }
        Ok(lanes.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spaces::{DType, SpaceValue, Tensor};

    fn obs_space() -> spaces::SpaceSpec {
        spaces::spaces::BoxSpaceBuilder::scalar(0.0, 255.0, vec![1])
            .dtype(DType::Uint8)
            .build()
            .unwrap()
    }

    fn box_u8(v: u8) -> SpaceValue {
        SpaceValue::Box(Tensor::from_vec(vec![v], vec![1], DType::Uint8).unwrap())
    }

    /// A `ModelObservation` carrying `values` encoded against `obs_space()`.
    fn observation(values: &[SpaceValue]) -> ModelObservation {
        let space = obs_space();
        let wire = rlmesh_grpc::wire::encode_batched_partial_values(values, &space).unwrap();
        let contract = spaces::EnvContract {
            id: "T".into(),
            action_space: None,
            observation_space: Some(space),
            metadata: None,
            render_mode: String::new(),
            num_envs: values.len() as u32,
            autoreset_mode: Default::default(),
        };
        ModelObservation {
            observation: Some(wire.leaves),
            route: ModelRouteContext::default(),
            reset: false,
            num_envs: values.len(),
            env_contract: Some(Arc::new(contract)),
        }
    }

    #[test]
    fn decoded_lanes_roundtrips_each_lane() {
        let lanes = vec![box_u8(5), box_u8(9), box_u8(0)];
        assert_eq!(observation(&lanes).decoded_lanes().unwrap(), lanes);
    }

    #[test]
    fn decoded_requires_exactly_one_lane() {
        // Single lane decodes to the value; multi-lane errors rather than
        // silently taking lane 0.
        assert_eq!(observation(&[box_u8(7)]).decoded().unwrap(), box_u8(7));
        assert!(observation(&[box_u8(1), box_u8(2)]).decoded().is_err());
    }

    #[test]
    fn absent_observation_errors_not_empty_vec() {
        // An absent observation must error, never return an empty Vec that would
        // silently satisfy a `== num_envs` check for num_envs == 0 callers.
        let mut obs = observation(&[box_u8(3)]);
        obs.observation = None;
        assert!(obs.decoded_lanes().is_err());
    }
}

/// Notification that an episode has ended, passed to
/// [`ModelHandler::on_episode_end`](crate::ModelHandler::on_episode_end).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelEpisodeEnd {
    /// Identifier of the episode that ended.
    pub episode_id: String,
    /// Index of the sub-environment that produced the episode.
    pub env_index: i32,
}

/// Notification that a single lane's episode rolled, passed to
/// [`ModelHandler::on_lane_reset`](crate::ModelHandler::on_lane_reset). Unlike
/// the whole-observation `on_reset`, this carries the `env_index` so a stateful
/// policy/adapter can reset exactly the lane whose episode began.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelLaneReset {
    /// Identifier of the new episode that began on this lane.
    pub episode_id: String,
    /// Index of the sub-environment whose episode rolled.
    pub env_index: i32,
}
