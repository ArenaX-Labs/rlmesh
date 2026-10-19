//! Native single-call entry points. Admission and subscriptions belong to the facade.
use crate::ext::CancelToken;
use crate::{InfoDict, RemoteEnv, RemoteVectorEnv, Result, Session, Value, VectorSeed};
/// Reset with an optional borrowed per-operation token.
pub fn session_reset(
    handle: &mut Session,
    seed: Option<i64>,
    trial_index: Option<i64>,
    cancel: Option<&CancelToken>,
) -> Result<(Value, InfoDict)> {
    handle.reset_cancel(seed, trial_index, cancel.cloned())
}
/// Step with an optional borrowed per-operation token.
#[allow(clippy::type_complexity)]
pub fn session_step(
    handle: &mut Session,
    action: Value,
    cancel: Option<&CancelToken>,
) -> Result<(Value, f64, bool, bool, InfoDict)> {
    handle.step_cancel(action, cancel.cloned())
}
/// Reset with an optional borrowed per-operation token.
pub fn remote_env_reset(
    handle: &mut RemoteEnv,
    seed: Option<i64>,
    options: Option<&InfoDict>,
    cancel: Option<&CancelToken>,
) -> Result<(Value, InfoDict)> {
    handle.reset_cancel(seed, options, cancel.cloned())
}
/// Step with an optional borrowed per-operation token.
#[allow(clippy::type_complexity)]
pub fn remote_env_step(
    handle: &mut RemoteEnv,
    action: Value,
    cancel: Option<&CancelToken>,
) -> Result<(Value, f64, bool, bool, InfoDict)> {
    handle.step_cancel(action, cancel.cloned())
}
/// Reset with an optional borrowed per-operation token.
pub fn remote_vector_env_reset(
    handle: &mut RemoteVectorEnv,
    seed: Option<VectorSeed>,
    options: Option<&InfoDict>,
    cancel: Option<&CancelToken>,
) -> Result<(Value, InfoDict)> {
    handle.reset_cancel(seed, options, cancel.cloned())
}
/// Step with an optional borrowed per-operation token.
#[allow(clippy::type_complexity)]
pub fn remote_vector_env_step(
    handle: &mut RemoteVectorEnv,
    action: Value,
    cancel: Option<&CancelToken>,
) -> Result<(Value, Vec<f64>, Vec<bool>, Vec<bool>, InfoDict)> {
    handle.step_cancel(action, cancel.cloned())
}
/// Predict with an optional borrowed per-operation token.
pub fn session_predict(
    handle: &mut Session,
    observation: Value,
    cancel: Option<&CancelToken>,
) -> Result<Value> {
    handle.predict_cancel(observation, cancel.cloned())
}
