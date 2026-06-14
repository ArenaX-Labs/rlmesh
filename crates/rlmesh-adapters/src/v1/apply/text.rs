//! Set one model text input on the payload, honoring the plan's default.

use std::collections::BTreeMap;

use super::super::plans::TextPlan;
use super::super::spec::TextContainer;
use super::lookup::lookup;
use super::value::Value;

/// Set one model text input on the payload, honoring the plan's default.
pub(super) fn apply_text(
    plan: &TextPlan,
    raw_obs: &BTreeMap<String, Value>,
    payload: &mut BTreeMap<String, Value>,
) {
    let mut value: Option<String> = None;
    if !plan.env_key.is_empty()
        && let Ok(found) = lookup(raw_obs, &plan.env_key)
    {
        value = Some(match found {
            Value::Text(text) => text.clone(),
            Value::Number(number) => format!("{number}"),
            other => format!("{other:?}"),
        });
    }
    let Some(value) = value.or_else(|| plan.default.clone()) else {
        return;
    };
    let entry = if plan.container == TextContainer::List {
        Value::List(vec![Value::Text(value)])
    } else {
        Value::Text(value)
    };
    payload.insert(plan.model_key.clone(), entry);
}
