//! Apply resolved observation plans: dispatch each model input to its
//! per-kind application (image / state / text / custom).

use std::collections::BTreeMap;

use super::super::plans::ObsPlan;
use super::CustomTransform;
use super::error::ApplyError;
use super::image::apply_image;
use super::state::apply_state;
use super::text::apply_text;
use super::value::Value;

/// Convert a raw env observation into the model input payload.
pub fn transform_obs(
    plans: &[ObsPlan],
    raw_obs: &BTreeMap<String, Value>,
    customs: &dyn CustomTransform,
) -> Result<BTreeMap<String, Value>, ApplyError> {
    let mut payload: BTreeMap<String, Value> = BTreeMap::new();
    for plan in plans {
        match plan {
            ObsPlan::Image(image_plan) => {
                payload.insert(
                    image_plan.model_key.clone(),
                    apply_image(image_plan, raw_obs)?,
                );
            }
            ObsPlan::State(state_plan) => {
                payload.insert(
                    state_plan.model_key.clone(),
                    apply_state(state_plan, raw_obs)?,
                );
            }
            ObsPlan::Text(text_plan) => apply_text(text_plan, raw_obs, &mut payload),
            ObsPlan::Custom(custom_plan) => {
                if let Some(value) =
                    customs.apply(&custom_plan.model_key, &custom_plan.transform, raw_obs)?
                {
                    payload.insert(custom_plan.model_key.clone(), value);
                }
            }
        }
    }
    Ok(payload)
}
