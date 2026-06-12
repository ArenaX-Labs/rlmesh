//! Resolved instructions for one model text input.

use super::super::spec::TextContainer;

/// Resolved instructions for one model text input.
#[derive(Debug, Clone, PartialEq)]
pub struct TextPlan {
    pub model_key: String,
    pub env_key: String,
    pub container: TextContainer,
    pub default: Option<String>,
}
