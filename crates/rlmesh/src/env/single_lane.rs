//! A one-lane vector env served as a scalar env.
//!
//! The contract's `num_envs` is the only vector signal on the wire, so a
//! batched simulator configured with one lane is served as a scalar lane:
//! the runtime then owns its resets, per-episode seeds and `trial_index`.

use rlmesh_spaces::SpaceSpec as WireSpec;

use super::batch::{batch1, first_item, info0, lane0};
use super::sync_wire::autoreset_mode;
use super::{Env, VectorEnv, VectorSeed};
use crate::spaces::Space;
use crate::{Error, InfoDict, Metadata, Result, Value};

const AUTORESET_KEY: &str = "autoreset_mode";

/// A scalar view of a vector env whose `num_envs` is 1.
pub(crate) struct SingleLaneEnv {
    env: Box<dyn VectorEnv + Send>,
    observation_space: Space,
    action_space: Space,
    metadata: Metadata,
}

impl SingleLaneEnv {
    /// Reject `SameStep` autoreset, validate the mode, then remove the
    /// `autoreset_mode` metadata key.
    pub(crate) fn new(env: Box<dyn VectorEnv + Send>) -> Result<SingleLaneEnv> {
        let mut metadata = env.metadata().unwrap_or_default();
        if metadata.get(AUTORESET_KEY) == Some(&Value::Text("SameStep".to_string())) {
            return Err(Error::invalid_argument(
                "a one-lane vector env with SAME_STEP autoreset returns the next episode's \
                 first observation in place of the terminal one; construct it with NEXT_STEP \
                 autoreset",
            ));
        }
        autoreset_mode(Some(&metadata))?;
        metadata.remove(AUTORESET_KEY);
        Ok(SingleLaneEnv {
            observation_space: env.single_observation_space(),
            action_space: env.single_action_space(),
            env,
            metadata,
        })
    }

    fn observation_wire(&self) -> &WireSpec {
        self.observation_space.spec().wire()
    }
}

fn lane_error(err: Error) -> Error {
    // A projection failure is the env's, as Python's `_lane0` raising inside
    // the callback is: not recoverable.
    Error::environment(err.to_string())
}

impl Env for SingleLaneEnv {
    fn observation_space(&self) -> Space {
        self.observation_space.clone()
    }

    fn action_space(&self) -> Space {
        self.action_space.clone()
    }

    fn reset(
        &mut self,
        seed: Option<i64>,
        options: Option<&InfoDict>,
    ) -> Result<(Value, InfoDict)> {
        let (observation, info) = self.env.reset(seed.map(VectorSeed::One), options)?;
        Ok((
            lane0(self.observation_wire(), observation).map_err(lane_error)?,
            info0(&info),
        ))
    }

    fn step(&mut self, action: Option<Value>) -> Result<(Value, f64, bool, bool, InfoDict)> {
        let space = self.action_space.spec().wire();
        let actions = match action {
            Some(action) => batch1(space, action).map_err(lane_error)?,
            None => match &space.spec {
                // Python's `_batch1` indexes the None: a TypeError inside step,
                // except a Dict with no children, which has nothing to index.
                Some(rlmesh_spaces::types::SpaceKind::Dict(dict)) if dict.keys.is_empty() => {
                    Value::Dict(Metadata::new())
                }
                Some(
                    rlmesh_spaces::types::SpaceKind::Dict(_)
                    | rlmesh_spaces::types::SpaceKind::Tuple(_),
                ) => {
                    return Err(Error::environment("'NoneType' object is not subscriptable"));
                }
                _ => Value::List(vec![Value::None]),
            },
        };
        let (observation, rewards, terminated, truncated, info) = self.env.step(actions)?;
        let first = |name: &str, len: usize| {
            (len > 0)
                .then_some(())
                .ok_or_else(|| Error::environment(format!("step returned no {name}")))
        };
        first("rewards", rewards.len())?;
        first("terminated values", terminated.len())?;
        first("truncated values", truncated.len())?;
        Ok((
            lane0(self.observation_wire(), observation).map_err(lane_error)?,
            rewards[0],
            terminated[0],
            truncated[0],
            info0(&info),
        ))
    }

    fn render(&mut self) -> Result<Option<Value>> {
        match self.env.render()? {
            Some(frames) => first_item(frames).map(Some).map_err(lane_error),
            None => Ok(None),
        }
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn render_mode(&self) -> Option<String> {
        self.env.render_mode()
    }

    fn metadata(&self) -> Option<Metadata> {
        Some(self.metadata.clone())
    }

    fn id(&self) -> Option<String> {
        self.env.id()
    }
}
