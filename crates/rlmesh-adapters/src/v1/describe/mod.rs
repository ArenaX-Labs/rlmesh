//! Render resolved plans as human-readable summaries.

mod action;
mod image;
mod state;
mod text;

use super::fmt::{quoted, quoted_range};
use super::plans::{ObsPlan, ResolvedAdapter};

pub use action::describe_segment;

/// Summarize how one model input is derived from the observation.
pub fn describe_obs_plan(plan: &ObsPlan) -> String {
    match plan {
        ObsPlan::Image(image) => image::describe_image(image),
        ObsPlan::State(state) => state::describe_state(state),
        ObsPlan::Text(text) => text::describe_text(text),
        ObsPlan::Custom(custom) => {
            format!("{} <- custom transform", quoted(&custom.model_key))
        }
    }
}

pub(super) fn describe_adapter(adapter: &ResolvedAdapter) -> String {
    let mut lines: Vec<String> = vec!["observation:".to_owned()];
    for plan in &adapter.obs_plans {
        lines.push(format!("  {}", describe_obs_plan(plan)));
    }
    lines.push("action:".to_owned());
    for segment in &adapter.action_plan.segments {
        lines.push(format!("  {}", describe_segment(segment)));
    }
    if let Some(clip) = adapter.action_plan.clip {
        lines.push(format!("  clip to {}", quoted_range(clip)));
    }
    lines.join("\n")
}
