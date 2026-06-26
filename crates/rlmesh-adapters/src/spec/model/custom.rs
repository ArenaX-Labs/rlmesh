//! An escape-hatch input computed by host-language code.

use serde::{Deserialize, Serialize};

/// An escape-hatch input computed by host-language code.
///
/// The core never evaluates the entrypoint: resolution produces a
/// [`crate::v1::CustomPlan`] hole that the binding materializes itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomInput {
    pub key: String,
    /// Opaque host reference, not a transform value the core runs: either a
    /// `module:callable` entrypoint or a `host:<key>` placeholder for an
    /// in-process callable. The binding materializes it; the wire key stays
    /// `transform` (frozen). The Python side names the entrypoint variant's
    /// field `entrypoint`, but it travels under this `transform` key.
    pub transform: String,
}
