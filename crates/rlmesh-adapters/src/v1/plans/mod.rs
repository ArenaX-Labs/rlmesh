//! Plan types: the resolved, already-validated adapter instructions.

mod action;
mod custom;
mod image;
mod state;
mod text;

use super::describe;

pub use action::{ActionPlan, ActionSegment};
pub use custom::CustomPlan;
pub use image::ImagePlan;
pub use state::{StatePiece, StatePlan};
pub use text::TextPlan;

/// Resolved instructions for one model input.
#[derive(Debug, Clone, PartialEq)]
pub enum ObsPlan {
    Image(ImagePlan),
    State(StatePlan),
    Text(TextPlan),
    Custom(CustomPlan),
}

/// A resolved env-to-model adapter: build instances with [`super::resolve`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAdapter {
    pub obs_plans: Vec<ObsPlan>,
    pub action_plan: ActionPlan,
}

impl ResolvedAdapter {
    /// Human-readable summary of the resolved transformations.
    ///
    /// Byte-identical to the reference implementation's `describe()`; the
    /// conformance vectors pin the exact text.
    pub fn describe(&self) -> String {
        describe::describe_adapter(self)
    }
}
