//! Set one model text input on the payload, honoring the plan's default.

use std::collections::BTreeMap;

use super::lookup::lookup;
use super::value::Value;
use crate::error::ApplyError;
use crate::plans::TextPlan;
use crate::spec::TextContainer;

/// Set one model text input on the payload, honoring the plan's default.
pub(super) fn apply_text(
    plan: &TextPlan,
    raw_obs: &BTreeMap<String, Value>,
    payload: &mut BTreeMap<String, Value>,
) -> Result<(), ApplyError> {
    let mut value: Option<String> = None;
    if !plan.env_key.is_empty()
        && let Ok(found) = lookup(raw_obs, &plan.env_key)
    {
        value = Some(match found {
            Value::Text(text) => text.clone(),
            Value::Number(number) => format!("{number}"),
            other => {
                return Err(ApplyError::new(format!(
                    "text input '{}' resolved to '{}', but {other:?} is not a string",
                    plan.model_key, plan.env_key
                )));
            }
        });
    }
    let Some(value) = value.or_else(|| plan.default.clone()) else {
        return Ok(());
    };
    let entry = if plan.container == TextContainer::List {
        Value::List(vec![Value::Text(value)])
    } else {
        Value::Text(value)
    };
    payload.insert(plan.model_key.clone(), entry);
    Ok(())
}
