use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// An escape-hatch input computed by host-language code.
///
/// The core never evaluates the entrypoint: resolution produces a
/// [`crate::v1::CustomPlan`] hole that the binding materializes. There is no
/// `key`: placement is the tree position this leaf sits at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Custom {
    /// Opaque host reference, not a transform value the core runs: either a
    /// `module:callable` entrypoint or a `host:<key>` placeholder for an
    /// in-process callable. Python's entrypoint variant names it `entrypoint`;
    /// on the wire it is always `transform`.
    pub transform: String,
    /// Unrecognized additive fields, retained for round-trip (see the strict-v1 publish gate).
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}
