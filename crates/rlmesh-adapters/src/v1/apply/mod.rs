//! Apply resolved plans to concrete observation and action values.
//!
//! The engine operates on its own [`Value`] model (typed arrays, text,
//! lists, nested maps); bridging host tensor types (numpy, the runtime's
//! native values) is binding-layer work. Custom inputs are host-language
//! holes: provide a [`CustomTransform`] to fill them, or use [`NoCustoms`]
//! when the specs are fully declarative.

mod action;
mod error;
mod geometry;
mod image;
mod lookup;
mod obs;
mod state;
mod text;
mod value;

use std::collections::BTreeMap;

pub use error::ApplyError;
pub use geometry::convert_rotation;
pub use value::{Array, ArrayData, Dtype, Value};

use super::plans::ResolvedAdapter;

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

impl ResolvedAdapter {
    /// Convert a raw env observation into the model input payload.
    pub fn transform_obs(
        &self,
        raw_obs: &BTreeMap<String, Value>,
        customs: &dyn CustomTransform,
    ) -> Result<BTreeMap<String, Value>, ApplyError> {
        obs::transform_obs(&self.obs_plans, raw_obs, customs)
    }

    /// Convert a model action output into the env action vector (float32).
    pub fn transform_action(&self, raw_action: &Value) -> Result<Array, ApplyError> {
        action::transform_action(&self.action_plan, raw_action)
    }
}
