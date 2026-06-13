//! Summarize a resolved text plan.

use super::super::fmt::quoted;
use super::super::plans::TextPlan;

pub(super) fn describe_text(plan: &TextPlan) -> String {
    let source = if plan.env_key.is_empty() {
        format!(
            "default {}",
            match &plan.default {
                Some(default) => quoted(default),
                None => "None".to_owned(),
            }
        )
    } else {
        quoted(&plan.env_key)
    };
    format!("{} <- text {}", quoted(&plan.model_key), source)
}
