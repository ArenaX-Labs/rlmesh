use std::sync::Arc;

use rlmesh_grpc::wire::Bytes;

use crate::spaces;
use crate::{Error, Result};

/// Routing metadata attached to a [`ModelObservation`].
///
/// Identifies the env (adapter) the request belongs to and the ordered per-row
/// episode ids — the self-describing batch. Row `i` of the batched observation
/// belongs to `episode_ids[i]`. There is no positional lane/slot concept; the
/// model keys all per-episode state by `episode_id`, never by position.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelRouteContext {
    /// Identifier of the session this request belongs to (correlation only).
    pub session_id: String,
    /// The connected env container (UUIDv7) this adapter serves. The routing key.
    pub env_id: String,
    /// Identifier of this individual request.
    pub request_id: String,
    /// Ordered per-row episode ids (length `== num_envs`).
    pub episode_ids: Vec<String>,
}

impl ModelRouteContext {
    /// The first row's episode id, or `""` if the batch is empty. The natural
    /// choice for single-env handlers.
    pub fn primary_episode_id(&self) -> &str {
        self.episode_ids.first().map(String::as_str).unwrap_or("")
    }

    /// The episode id of every row, in batch order.
    pub fn episode_ids(&self) -> Vec<String> {
        self.episode_ids.clone()
    }
}

/// An observation delivered to [`ModelHandler::predict`](crate::ModelHandler::predict).
///
/// Carries the encoded observation payload plus the routing context and the
/// env contract / batch size needed to decode it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelObservation {
    /// Raw per-leaf observation wire bytes, if present. `None` is the only
    /// "absent value" signal (§5) — decode via
    /// [`decoded`](ModelObservation::decoded)/[`decoded_lanes`](ModelObservation::decoded_lanes),
    /// never read these bytes directly.
    pub observation: Option<Vec<Bytes>>,
    /// Routing/episode metadata for this request.
    pub route: ModelRouteContext,
    /// Number of rows (sub-environments) in this batch (`== route.episode_ids.len()`).
    pub num_envs: usize,
    /// The env contract (spaces/metadata) for decoding the observation. Shared
    /// (`Arc`) so the per-predict hot path clones a refcount, not the contract.
    pub env_contract: Option<Arc<spaces::EnvContract>>,
}

impl ModelObservation {
    /// The first row's episode id (see [`ModelRouteContext::primary_episode_id`]).
    pub fn episode_id(&self) -> &str {
        self.route.primary_episode_id()
    }

    /// The episode id of every row (see [`ModelRouteContext::episode_ids`]).
    pub fn episode_ids(&self) -> Vec<String> {
        self.route.episode_ids()
    }

    /// Validate the request carries everything a decode needs (observation, env
    /// contract, observation space) WITHOUT paying the batch decode. Lets a
    /// malformed request fail fast even on a pure chunk-replay step that skips the
    /// actual decode below, instead of silently replaying buffered actions.
    pub fn ensure_decodable(&self) -> Result<()> {
        if self.observation.is_none() {
            return Err(Error::model(
                "observation absent; a predict request must carry an observation",
            ));
        }
        let contract = self
            .env_contract
            .as_ref()
            .ok_or_else(|| Error::model("observation missing env contract; cannot decode"))?;
        if contract.observation_space.is_none() {
            return Err(Error::model("env contract missing observation space"));
        }
        Ok(())
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
    fn ensure_decodable_catches_malformed_requests_without_decoding() {
        // Happy path: present observation + contract + space validates.
        assert!(observation(&[box_u8(3)]).ensure_decodable().is_ok());
        // Absent observation errors up front.
        let mut no_obs = observation(&[box_u8(3)]);
        no_obs.observation = None;
        assert!(no_obs.ensure_decodable().is_err());
        // Missing env contract errors up front -- the gap that used to be deferred
        // on a step where every lane is mid-chunk-replay and skips the decode.
        let mut no_contract = observation(&[box_u8(3)]);
        no_contract.env_contract = None;
        assert!(no_contract.ensure_decodable().is_err());
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

