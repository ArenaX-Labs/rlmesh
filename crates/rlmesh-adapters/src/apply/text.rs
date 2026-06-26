//! Produce one model text input value, honoring the plan's default.

use std::collections::BTreeMap;

use super::lookup::resolve_in_obs;
use super::value::Value;
use crate::error::ApplyError;
use crate::plans::TextPlan;
use crate::spec::TextContainer;

/// Produce one model text input value, honoring the plan's default.
///
/// Returns `Ok(None)` when there is no env source value and no default — the
/// caller then omits the placement entirely (a text input the env never
/// provided and the model did not default).
pub(super) fn apply_text(
    plan: &TextPlan,
    raw_obs: &BTreeMap<String, Value>,
) -> Result<Option<Value>, ApplyError> {
    let mut value: Option<String> = None;
    if let Some(source) = &plan.source
        && let Ok(found) = resolve_in_obs(raw_obs, source)
    {
        value = Some(match found {
            Value::Text(text) => text.clone(),
            // A text input is only ever paired with an env Text feature, so a
            // non-string here means the env violated its declared space. Surface
            // it rather than silently stringifying a stray number.
            other => {
                return Err(ApplyError::new(format!(
                    "text input '{}' resolved to '{}', but {other:?} is not a string",
                    plan.placement, source
                )));
            }
        });
    }
    let Some(value) = value.or_else(|| plan.default.clone()) else {
        return Ok(None);
    };
    let entry = if plan.container == TextContainer::List {
        Value::List(vec![Value::Text(value)])
    } else {
        Value::Text(value)
    };
    Ok(Some(entry))
}
