//! An escape-hatch input computed by host-language code.

use serde::{Deserialize, Serialize};

/// An escape-hatch input computed by host-language code.
///
/// The core never evaluates the entrypoint: resolution produces a
/// [`crate::v1::CustomPlan`] hole that the binding materializes itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomInput {
    pub key: String,
    pub transform: String,
}
