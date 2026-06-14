//! Resolved hole for a custom-transform input.

/// Resolved hole for a custom-transform input.
///
/// The core never imports or evaluates the entrypoint; the host binding
/// materializes it (subject to its own trust policy).
#[derive(Debug, Clone, PartialEq)]
pub struct CustomPlan {
    pub model_key: String,
    pub transform: String,
}
