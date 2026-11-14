//! Run a binding's plans on host values: the observation and action
//! [`Value`] trees, Gym spaces, the JPEG codec and the host-language holes.
//!
//! The binder ([`rlmesh_session_core::spec::role`]) decides and converts;
//! this module reads each plan's sources out of the raw observation,
//! hands them over, and places what comes back into the payload tree.
//! Custom inputs are host-language holes: provide a [`CustomTransform`] to
//! fill them, or use [`NoCustoms`] when the specs are fully declarative.

mod jpeg;
pub(crate) mod lookup;
pub(crate) mod obs;
mod text;
pub(crate) mod value;

use std::collections::BTreeMap;

use rlmesh_session_core::spec::apply::Tensor as Array;
use rlmesh_spaces::{DType, Tensor};

pub use crate::error::ApplyError;
pub use jpeg::Jpeg;
pub use value::Value;

use crate::plans::Binding as ResolvedAdapter;

/// The resample algorithms an `Image` input may name. Un-suffixed names have
/// cv2/torch semantics, `_aa` names PIL's; bare `bicubic`/`lanczos3` are
/// deliberately absent, so a spec naming one fails resolution rather than
/// silently getting the other library's kernel.
pub const RESAMPLES: [&str; 5] = [
    "bilinear",
    "bilinear_aa",
    "bicubic_aa",
    "lanczos3_aa",
    "area",
];

/// Host-language hook materializing custom-input transforms.
pub trait CustomTransform {
    /// Produce the payload value for one custom input, or `Ok(None)` to
    /// omit the key (the host fills it outside the engine, e.g. a binding
    /// running the user's callable on the raw host-language observation).
    fn apply(
        &self,
        model_key: &str,
        entrypoint: &str,
        raw_obs: &BTreeMap<String, Value>,
    ) -> Result<Option<Value>, ApplyError>;
}

/// Rejects all custom inputs; use when specs are fully declarative.
pub struct NoCustoms;

impl CustomTransform for NoCustoms {
    fn apply(
        &self,
        model_key: &str,
        _entrypoint: &str,
        _raw_obs: &BTreeMap<String, Value>,
    ) -> Result<Option<Value>, ApplyError> {
        Err(ApplyError::new(format!(
            "custom input '{model_key}' requires a host-language transform; \
             provide a CustomTransform implementation"
        )))
    }
}

/// Omits custom inputs from the payload; the host fills them afterwards.
pub struct SkipCustoms;

impl CustomTransform for SkipCustoms {
    fn apply(
        &self,
        _model_key: &str,
        _entrypoint: &str,
        _raw_obs: &BTreeMap<String, Value>,
    ) -> Result<Option<Value>, ApplyError> {
        Ok(None)
    }
}

/// Convert a raw env observation into the model input payload `Value` tree.
///
/// The stateless half: a frame-stacked input yields its single frame and an
/// action-source part its `fill`. The per-episode windows live behind
/// [`assemble_obs`](crate::stateful::assemble_obs).
pub fn transform_obs(
    adapter: &ResolvedAdapter,
    raw_obs: &BTreeMap<String, Value>,
    customs: &dyn CustomTransform,
) -> Result<Value, ApplyError> {
    obs::transform_obs(&adapter.obs_plans, raw_obs, customs, None)
}

/// Convert a model action output into the env action vector (float32).
pub fn transform_action(
    adapter: &ResolvedAdapter,
    raw_action: &Value,
) -> Result<Tensor, ApplyError> {
    let action = lookup::numeric_vector(raw_action)?;
    let env = adapter.action_plan.apply(&action).map_err(from_core)?;
    Ok(value::tensor_from_f32(vec![env.len() as i64], &env))
}

/// A binder apply failure as the host's.
pub(crate) fn from_core(error: rlmesh_session_core::spec::apply::ApplyError) -> ApplyError {
    ApplyError::new(error.0)
}

/// A host failure as the binder's (a host read inside a plan).
pub(crate) fn to_core(error: ApplyError) -> rlmesh_session_core::spec::apply::ApplyError {
    rlmesh_session_core::spec::apply::ApplyError(error.message)
}

/// A binder array as a host tensor.
pub(crate) fn from_array(array: Array) -> Result<Tensor, ApplyError> {
    let dtype = DType::try_from(array.dtype.code() as i32).map_err(ApplyError::new)?;
    let shape = array.shape.iter().map(|&d| i64::from(d)).collect();
    Tensor::from_vec(array.data, shape, dtype).map_err(|err| ApplyError::new(err.to_string()))
}
