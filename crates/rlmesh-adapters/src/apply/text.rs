use std::collections::BTreeMap;

use super::lookup::resolve_in_obs;
use super::value::Value;
use crate::error::ApplyError;
use crate::plans::TextPlan;

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
            other => {
                return Err(ApplyError::new(format!(
                    "text input '{}' resolved to '{}', but {other:?} is not a string",
                    plan.placement, source
                )));
            }
        });
    }
    let Some(value) = value.or_else(|| plan.fill.clone()) else {
        return Ok(None);
    };
    let entry = if plan.list {
        Value::List(vec![Value::Text(value)])
    } else {
        Value::Text(value)
    };
    Ok(Some(entry))
}
