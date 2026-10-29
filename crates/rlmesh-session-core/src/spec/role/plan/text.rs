//! Resolved instructions for one model text input.

#[allow(unused_imports)]
use crate::prelude::*;

use crate::spec::role::path::NodePath;

/// Resolved instructions for one model text input.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPlan {
    /// Where this text lands in the assembled payload tree.
    pub placement: NodePath,
    /// Where this text is read from in the raw observation tree, or `None` for a
    /// fill-only text input that never looks one up (the env had no matching
    /// text role but the model declared a `fill`).
    pub source: Option<NodePath>,
    /// Delivered as a one-item list rather than a string.
    pub list: bool,
    /// Delivered when the env has no such text.
    pub fill: Option<String>,
}
