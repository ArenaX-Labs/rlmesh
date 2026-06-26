//! Resolved instructions for one model text input.

use crate::path::NodePath;
use crate::spec::TextContainer;

/// Resolved instructions for one model text input.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPlan {
    /// Where this text lands in the assembled payload tree.
    pub placement: NodePath,
    /// Where this text is read from in the raw observation tree, or `None` for a
    /// default-only text input that never looks one up (the env had no matching
    /// text role but the model declared a `default`).
    pub source: Option<NodePath>,
    pub container: TextContainer,
    pub default: Option<String>,
}
