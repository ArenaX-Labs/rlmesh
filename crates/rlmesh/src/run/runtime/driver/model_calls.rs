//! The driver's calls into the model handler, each under its lock: grouped
//! predict, reset options, route release and eviction.

use std::sync::Arc;
use std::time::Instant;

use rlmesh_proto::EndpointPhases;

use super::{Prediction, RuntimeError};
use crate::model::handler::{ModelHandler, ModelResetOptions, PredictFrames};
use crate::model::types::{EpisodeInfo, ModelObservation};

/// A model error as the driver's, keeping its recoverable flag.
/// The options the model adds for the lanes `starts` a reset starts as
/// `trials`: one lane's map, or for several lanes each key as a per-lane list
/// (`None` where a lane did not set it).
pub(super) async fn model_reset_options(
    source: Option<Arc<dyn ModelResetOptions>>,
    starts: &[EpisodeInfo],
    trials: &[u64],
) -> Result<Option<rlmesh_spaces::MetaMap>, RuntimeError> {
    use rlmesh_spaces::MetaValue;
    let Some(source) = source else {
        return Ok(None);
    };
    let mut lanes = Vec::with_capacity(starts.len());
    for (lane, episode) in starts.iter().enumerate() {
        let options = source
            .reset_options(episode, trials.get(lane).copied())
            .await
            .map_err(model_error)?;
        lanes.push(options.unwrap_or_default());
    }
    if let [only] = lanes.as_mut_slice() {
        return Ok(Some(std::mem::take(only)));
    }
    let keys: std::collections::BTreeSet<&String> = lanes.iter().flat_map(|l| l.keys()).collect();
    Ok(Some(
        keys.into_iter()
            .map(|key| {
                let values = lanes
                    .iter()
                    .map(|lane| lane.get(key).cloned().unwrap_or(MetaValue::Null))
                    .collect();
                (key.clone(), MetaValue::List(values))
            })
            .collect(),
    ))
}

/// The model's reset options under the runtime's own reserved ones.
pub(super) fn merge_options(
    model: Option<rlmesh_spaces::MetaMap>,
    reserved: Option<rlmesh_spaces::MetaMap>,
) -> Option<rlmesh_spaces::MetaMap> {
    let mut options = model.unwrap_or_default();
    options.extend(reserved.unwrap_or_default());
    (!options.is_empty()).then_some(options)
}

fn model_error(error: crate::Error) -> RuntimeError {
    RuntimeError::model_rpc_with_recoverability("model", error.is_transient(), error)
}

/// One grouped predict: every observation in one `predict_grouped` call, the
/// model's fusion seam, then each group's frames encoded against the action
/// space. One group's failure is its own.
pub(super) async fn predict_grouped<M: ModelHandler>(
    model: &tokio::sync::Mutex<M>,
    contract: &rlmesh_spaces::EnvContract,
    observations: Vec<ModelObservation>,
) -> Vec<Result<Prediction, RuntimeError>> {
    let started = Instant::now();
    let Some(action_space) = contract.action_space.as_ref() else {
        return observations
            .iter()
            .map(|_| {
                Err(model_error(crate::Error::model(
                    "the env contract has no action space",
                )))
            })
            .collect();
    };
    let rows: Vec<usize> = observations.iter().map(|o| o.num_envs).collect();
    let mut handler = model.lock().await;
    let call_started = Instant::now();
    let results = handler.predict_grouped(observations).await;
    let user_ns = rlmesh_proto::elapsed_ns(call_started);
    // Drain the adapter share even for a failed forward, or its time leaks
    // into the next predict's.
    let adapter_ns = handler.take_adapter_ns();
    let held = handler.held_state();
    // A served model's own timing replaces this call's: its user time is its
    // share of a forward it may have fused with other routes' requests, and
    // the call's wall time is the op's RPC.
    let (endpoint_total_ns, endpoint) = match handler.take_endpoint_timing() {
        Some((total, phases)) => (total, Some(phases)),
        None => (None, None),
    };
    drop(handler);
    let encode_started = Instant::now();
    let mut results = results.into_iter();
    rows.into_iter()
        .map(|rows| {
            let PredictFrames { actions, replay } = results
                .next()
                .ok_or_else(|| {
                    model_error(crate::Error::model(
                        "predict_grouped returned fewer results than groups",
                    ))
                })?
                .map_err(model_error)?;
            let frames = std::iter::once(actions)
                .chain(replay)
                .map(|frame| {
                    if frame.len() != rows {
                        return Err(model_error(crate::Error::model(format!(
                            "predict returned {} actions for {rows} lanes",
                            frame.len()
                        ))));
                    }
                    crate::model::types::check_actions_conform(action_space, &frame)
                        .map_err(model_error)?;
                    rlmesh_spaces::leaves::encode_leaf_slab(&frame, action_space)
                        .map_err(|err| model_error(crate::Error::model(err.to_string())))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let phases = endpoint.unwrap_or(EndpointPhases {
                user_ns,
                encode_ns: rlmesh_proto::elapsed_ns(encode_started),
                ..EndpointPhases::default()
            });
            Ok(Prediction {
                frames,
                // Timed here only when the model reported nothing: a total
                // it left out stays absent beside the phases it reported.
                endpoint_total_ns: if endpoint.is_some() {
                    endpoint_total_ns
                } else {
                    Some(rlmesh_proto::elapsed_ns(started))
                },
                phases: EndpointPhases {
                    adapter_ns,
                    held_episodes: held.map(|held| held.episodes.min(u64::from(u32::MAX)) as u32),
                    held_state_bytes: held.map(|held| held.bytes),
                    ..phases
                },
            })
        })
        .collect()
}

/// Drops the model's route state at route end.
pub(super) async fn release<M: ModelHandler>(
    model: &tokio::sync::Mutex<M>,
    env_id: String,
) -> Result<(), RuntimeError> {
    let route_setup = model.lock().await.route_setup();
    match route_setup {
        Some(route_setup) => route_setup
            .release_adapter(&env_id)
            .await
            .map_err(model_error),
        None => Ok(()),
    }
}

/// Ends `episode_ids` on the model: the route's own state first, then the
/// model's hook, as a served model does.
pub(super) async fn evict<M: ModelHandler>(
    model: &tokio::sync::Mutex<M>,
    env_id: String,
    episode_ids: Vec<String>,
) -> Result<(), RuntimeError> {
    let mut handler = model.lock().await;
    if let Some(route_setup) = handler.route_setup() {
        route_setup
            .reset_adapter(&env_id, &episode_ids)
            .await
            .map_err(model_error)?;
    }
    handler
        .reset_adapter(&env_id, episode_ids)
        .await
        .map_err(model_error)
}
