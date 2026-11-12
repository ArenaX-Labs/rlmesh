//! The link a client drives its env through: the lockstep exchange or a Gym
//! view, and how its failures read.

use std::sync::Arc;
use std::time::Duration;

use rlmesh_spaces::{MetaMap, MetaValue};

use super::episodes::Episodes;
use crate::env::types::StepResult as VectorStepResult;
use crate::{Error, Result, binding::spaces};

/// A contract's observation and action spaces; one without them is a
/// malformed handshake.
pub(super) fn spaces_of(
    contract: &spaces::EnvContract,
) -> Result<(Arc<spaces::SpaceSpec>, Arc<spaces::SpaceSpec>)> {
    match (&contract.observation_space, &contract.action_space) {
        (Some(observation), Some(action)) => {
            Ok((Arc::new(observation.clone()), Arc::new(action.clone())))
        }
        _ => Err(Error::Protocol(
            "remote environment contract is missing its spaces".to_string(),
        )),
    }
}

/// A lane index for the wire (`uint32`): above `u32::MAX` is the ValueError
/// class, before any I/O.
pub(super) fn wire_lane(index: u64) -> Result<u32> {
    u32::try_from(index).map_err(|_| {
        Error::invalid_argument(format!(
            "env_index {index} is out of range: lane indices are at most {}",
            u32::MAX
        ))
    })
}

/// A shared read handle on a client's selected workflow edition: the
/// handshake selection until a pin replaces it.
#[derive(Clone, Debug)]
pub struct EditionCell(pub(super) Edition);

#[derive(Clone, Debug)]
pub(super) enum Edition {
    Local(Arc<std::sync::Mutex<String>>),
}

impl EditionCell {
    /// The current selection.
    pub fn get(&self) -> String {
        match &self.0 {
            Edition::Local(edition) => edition
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone(),
        }
    }
}

/// A session participant driver or a native robot view.
pub(super) enum Link {
    Lockstep {
        client: crate::lockstep::LockstepClient,
        address: String,
        /// The selection; lockstep sends no pin (the env runs its declared
        /// edition), so a pin is only recorded.
        edition: Arc<std::sync::Mutex<String>>,
        episodes: Episodes,
    },
    /// A v2-native lockstep robot, through its Gym view (one lane).
    View {
        /// `None` once closed (the lease is released).
        view: Arc<std::sync::Mutex<Option<Arc<crate::lockstep::GymView>>>>,
        address: String,
        edition: Arc<std::sync::Mutex<String>>,
    },
}

/// The Gym view behind a [`Link::View`], unless closed.
fn open_view(
    view: &std::sync::Mutex<Option<Arc<crate::lockstep::GymView>>>,
) -> Result<Arc<crate::lockstep::GymView>> {
    view.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
        .ok_or_else(|| Error::Runtime("the env client is closed".into()))
}

/// How a session host's lease refusal reads: another driver holds the env.
pub(crate) const ENV_HELD: &str = "the env is held by another driver (a binding, a run or a \
     RemoteEnv): close that one first";

/// Whether the host refused a lease another driver holds.
pub(crate) fn env_held(error: &crate::lockstep::GymError) -> bool {
    matches!(
        error,
        crate::lockstep::GymError::Session(rlmesh_session::SessionError::Rejected(refusal))
            if refusal.status == rlmesh_session::wire::CtlStatus::LeaseBusy
    )
}

pub(crate) fn lockstep_error(error: crate::lockstep::GymError) -> Error {
    use crate::lockstep::GymError;
    match error {
        error if env_held(&error) => Error::Connection(ENV_HELD.into()),
        GymError::Timeout(timeout) => Error::Timeout(timeout),
        GymError::Closed(why) => Error::Connection(why),
        // A refusal the same call meets again (a key the host does not
        // accept): final, so not the transient ConnectionError class.
        error @ GymError::Session(_) if !error.is_transient() => Error::Protocol(error.to_string()),
        GymError::Session(error) => Error::Connection(error.to_string()),
        GymError::Map(why) | GymError::Config(why) | GymError::Value(why) => {
            Error::invalid_argument(why)
        }
        other => Error::Environment(other.to_failure()),
    }
}

impl Link {
    pub(super) fn address(&self) -> &str {
        match self {
            Self::Lockstep { address, .. } | Self::View { address, .. } => address,
        }
    }

    /// Resets `lanes` (empty: every lane): the observations and the info.
    pub(super) async fn reset(
        &mut self,
        lanes: Vec<u32>,
        seeds: Vec<i64>,
        options: Option<MetaMap>,
        episode_ids: Vec<String>,
        timeout: Option<Duration>,
    ) -> Result<(Vec<spaces::SpaceValue>, Option<MetaMap>)> {
        match self {
            Self::Lockstep {
                client, episodes, ..
            } => {
                let reset = client
                    .reset(&lanes, &seeds, options.as_ref(), timeout)
                    .await
                    .map_err(lockstep_error)?;
                episodes.reset(&lanes, &seeds, &episode_ids);
                Ok((reset.observations, reset.info))
            }
            Self::View { view, .. } => {
                if !lanes.is_empty() {
                    return Err(Error::invalid_argument("a Gym view has one lane"));
                }
                // A v2-native robot reads plain JSON options.
                let options = options.map(|options| {
                    let lowered = MetaValue::Map(options).lowered();
                    let MetaValue::Map(options) = lowered else {
                        unreachable!("a map lowers to a map")
                    };
                    crate::lockstep::meta_json::to_json(&options)
                });
                let step = open_view(view)?
                    .reset(
                        seeds.first().map(|s| *s as u64),
                        &options.unwrap_or_default(),
                    )
                    .await
                    .map_err(lockstep_error)?;
                view_outcome(step).map(|(observation, info)| (vec![observation], Some(info)))
            }
        }
    }

    /// Steps every lane with one action each.
    pub(super) async fn step(
        &mut self,
        actions: &[spaces::SpaceValue],
        episode_ids: Vec<String>,
        timeout: Option<Duration>,
    ) -> Result<VectorStepResult> {
        match self {
            Self::Lockstep {
                client, episodes, ..
            } => {
                let step = client
                    .step(&[], actions, timeout)
                    .await
                    .map_err(lockstep_error)?;
                let mut result = VectorStepResult {
                    observations: step.observations,
                    rewards: step.rewards,
                    terminated: step.terminated,
                    truncated: step.truncated,
                    info: step.info,
                    completed_episodes: Vec::new(),
                    episode_ids: Vec::new(),
                };
                let mode = client.info().contract.autoreset_mode;
                result.completed_episodes = episodes.step(&result, &episode_ids, mode);
                Ok(result)
            }
            Self::View { view, .. } => {
                let [action] = actions else {
                    return Err(Error::invalid_argument("a Gym view has one lane"));
                };
                let actions =
                    crate::lockstep::view::action_values(action).map_err(lockstep_error)?;
                let step = open_view(view)?
                    .step(&actions)
                    .await
                    .map_err(lockstep_error)?;
                let (reward, terminated, truncated) =
                    (step.reward, step.terminated, step.truncated);
                let (observation, info) = view_outcome(step)?;
                Ok(VectorStepResult {
                    observations: vec![observation],
                    rewards: vec![reward],
                    terminated: vec![terminated],
                    truncated: vec![truncated],
                    info: Some(info),
                    completed_episodes: Vec::new(),
                    episode_ids: Vec::new(),
                })
            }
        }
    }

    /// Lane `lane`'s frame as PNG bytes, if it has one.
    pub(super) async fn render(
        &mut self,
        lane: u32,
        timeout: Option<Duration>,
    ) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Lockstep { client, .. } => client
                .render(lane, timeout)
                .await
                .map(|frame| frame.map(|frame| frame.to_vec()))
                .map_err(lockstep_error),
            Self::View { view, .. } => open_view(view)?
                .render()
                .await
                .map(|frame| frame.map(|frame| frame.to_vec()))
                .map_err(lockstep_error),
        }
    }
}

/// A view step's observation and info.
fn view_outcome(step: crate::lockstep::view::ViewStep) -> Result<(spaces::SpaceValue, MetaMap)> {
    let info = crate::lockstep::view::info(&step).map_err(lockstep_error)?;
    let observation =
        crate::lockstep::view::observation_value(step.observation).map_err(lockstep_error)?;
    Ok((observation, info))
}
