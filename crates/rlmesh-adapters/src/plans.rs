//! Plan types: the resolved, already-validated adapter instructions.

mod action;
mod custom;
mod image;
mod state;
mod text;

use std::collections::{BTreeMap, BTreeSet};

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

/// A resolved env-to-model adapter: build instances with [`resolve`](crate::resolver::resolve).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedAdapter {
    pub obs_plans: Vec<ObsPlan>,
    pub action_plan: ActionPlan,
}

impl ResolvedAdapter {
    /// Human-readable summary of the resolved transformations.
    ///
    /// A *reference snapshot*: the conformance vectors pin the exact text so
    /// implementations stay consistent, but the wording is not a stable
    /// cross-language contract.
    pub fn describe(&self) -> String {
        describe::describe_adapter(self)
    }

    /// Per-env data-loss / fabrication notes (a zero-filled camera, an aspect
    /// crop or letterbox): the "warn" subset of [`describe`](Self::describe),
    /// for a caller to surface (e.g. log) without failing. Empty when nothing
    /// noteworthy happened.
    pub fn advisories(&self) -> Vec<String> {
        describe::adapter_advisories(self)
    }

    /// The observation keys this adapter actually reads.
    ///
    /// Lets a caller decode only the referenced observation leaves instead of
    /// the whole observation — so an unused (possibly unencodable) env key
    /// never aborts a step. Custom inputs are excluded: they resolve against
    /// the raw host observation, not the decoded payload.
    pub fn referenced_obs_keys(&self) -> BTreeSet<String> {
        let mut keys = BTreeSet::new();
        for plan in &self.obs_plans {
            match plan {
                ObsPlan::Image(image) => {
                    // A zero-filled image has no env source (empty env_key).
                    if image.zero_fill.is_none() {
                        keys.insert(image.env_key.clone());
                    }
                }
                ObsPlan::State(state) => {
                    for piece in &state.pieces {
                        if !piece.zero_fill {
                            keys.insert(piece.env_key.clone());
                        }
                    }
                }
                ObsPlan::Text(text) => {
                    // A default-only text input has no env key (it never looks
                    // one up); reporting "" would make a caller try to encode a
                    // non-existent top-level key.
                    if !text.env_key.is_empty() {
                        keys.insert(text.env_key.clone());
                    }
                }
                ObsPlan::Custom(_) => {}
            }
        }
        keys
    }

    /// Frame-stack depths the model wants, keyed by model input key.
    ///
    /// Only entries with depth `> 1` (actual stacking) appear — a `stack == 1`
    /// image needs no per-episode buffer and is omitted. The episode-keyed
    /// frame-stack engine ([`crate::v1::FrameBuffers`]) buffers exactly these keys.
    /// This is the single source of truth for stacking depth (replacing the old
    /// Python `stacks` dict).
    pub fn stacks(&self) -> BTreeMap<String, u32> {
        let mut stacks = BTreeMap::new();
        for plan in &self.obs_plans {
            if let ObsPlan::Image(image) = plan
                && image.stack > 1
            {
                stacks.insert(image.model_key.clone(), image.stack);
            }
        }
        stacks
    }
}
