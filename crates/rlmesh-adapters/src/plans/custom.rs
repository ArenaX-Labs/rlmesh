//! Resolved hole for a custom-transform input.

/// Resolved hole for a custom-transform input.
///
/// The core never imports or evaluates the entrypoint; the host binding
/// materializes it (subject to its own trust policy).
#[derive(Debug, Clone, PartialEq)]
pub struct CustomPlan {
    pub model_key: String,
    /// Opaque host reference (a `module:callable` entrypoint or `host:<key>`
    /// placeholder), not a transform value the core runs; the binding
    /// materializes it. Carried verbatim from `CustomInput::transform`.
    pub transform: String,
}
