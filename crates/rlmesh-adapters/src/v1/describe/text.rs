//! Summarize a resolved text plan.

use super::super::plans::TextPlan;
use super::super::pyfmt::py_repr;

pub(super) fn describe_text(plan: &TextPlan) -> String {
    let source = if plan.env_key.is_empty() {
        format!(
            "default {}",
            match &plan.default {
                Some(default) => py_repr(default),
                None => "None".to_owned(),
            }
        )
    } else {
        py_repr(&plan.env_key)
    };
    format!("{} <- text {}", py_repr(&plan.model_key), source)
}
