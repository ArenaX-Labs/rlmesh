//! Generated RLMesh protobuf bindings and protocol-level constants.

use std::collections::HashMap;

/// Current RLMesh protocol generation.
///
/// This is intentionally not the package version. Package patch releases can
/// move independently when the wire contract remains compatible.
pub const PROTOCOL_GENERATION: &str = "rlmesh.protocol.v1";

/// Oldest protocol generation accepted by this crate.
pub const MIN_SUPPORTED_PROTOCOL_GENERATION: &str = "rlmesh.protocol.v1";

/// Current workflow semantics edition.
pub const CURRENT_WORKFLOW_EDITION: &str = "2026.06";

/// Workflow editions this crate can operate under. Each edition names an
/// immutable behavioral contract documented in `docs/editions/<edition>.md`.
pub const SUPPORTED_WORKFLOW_EDITIONS: &[&str] = &[CURRENT_WORKFLOW_EDITION];

/// Stable capability names exchanged during handshake.
pub mod capabilities {
    /// Core environment handshake and Join stream.
    pub const ENV_SERVICE_V1: &str = "rlmesh.env.service.v1";

    /// Core model handshake and Join stream.
    pub const MODEL_SERVICE_V1: &str = "rlmesh.model.service.v1";

    /// Core RLMesh space specifications and values.
    pub const SPACES_CORE_V1: &str = "rlmesh.spaces.core.v1";

    /// A served model endpoint processes Join-stream requests concurrently
    /// (pipelined predict): responses arrive in completion order rather than
    /// strict arrival order, while per-route lifecycle ordering is preserved.
    ///
    /// Advisory only and never an edition change — the wire messages are
    /// identical (every response still mirrors its `request_id`). A client uses
    /// it to decide whether overlapping multiple predicts on one connection will
    /// actually pipeline (capability present) or serialize behind the handler
    /// (capability absent). See `docs/editions/2026.06.md`.
    pub const MODEL_CONCURRENT_PREDICT_V1: &str = "rlmesh.model.concurrent_predict.v1";
}

/// Return whether a client protocol generation can speak to a server protocol generation.
pub fn is_protocol_generation_compatible(client: &str, server: &str) -> bool {
    client.trim() == PROTOCOL_GENERATION && server.trim() == PROTOCOL_GENERATION
}

/// Return whether a workflow edition is supported.
pub fn is_workflow_edition_supported(edition: &str) -> bool {
    SUPPORTED_WORKFLOW_EDITIONS.contains(&edition.trim())
}

/// Select the workflow edition governing a session from a peer's offer.
///
/// Returns the highest edition both peers support; the zero-padded `YYYY.MM`
/// format makes lexicographic order chronological. `None` means there is no
/// mutual edition and the handshake must report `compatible = false`. Only
/// explicitly supported editions are eligible — unknown editions in the offer
/// are ignored, never accepted on the assumption they are compatible.
pub fn negotiate_workflow_edition(offered: &[String]) -> Option<&'static str> {
    SUPPORTED_WORKFLOW_EDITIONS
        .iter()
        .copied()
        .filter(|edition| offered.iter().any(|offer| offer.trim() == *edition))
        .max()
}

/// Return supported workflow editions as owned strings for protobuf messages.
pub fn supported_workflow_editions() -> Vec<String> {
    SUPPORTED_WORKFLOW_EDITIONS
        .iter()
        .map(|edition| (*edition).to_string())
        .collect()
}

/// Return a handshake capability map for the given capability names.
pub fn capability_map(names: &[&str]) -> HashMap<String, String> {
    names
        .iter()
        .map(|name| ((*name).to_string(), "true".to_string()))
        .collect()
}

/// Return missing required capability names.
pub fn missing_required_capabilities<'a>(
    required: &[&'a str],
    offered: &HashMap<String, String>,
) -> Vec<&'a str> {
    required
        .iter()
        .copied()
        .filter(|name| !offered.contains_key(*name))
        .collect()
}

pub mod common {
    pub mod v1 {
        tonic::include_proto!("rlmesh.common.v1");
    }
}

pub mod core {
    pub mod v1 {
        tonic::include_proto!("rlmesh.core.v1");
    }
}

pub mod env {
    pub mod v1 {
        tonic::include_proto!("rlmesh.env.v1");
    }
}

pub mod spaces {
    pub mod v1 {
        tonic::include_proto!("rlmesh.spaces.v1");
    }
}

pub mod model {
    pub mod v1 {
        tonic::include_proto!("rlmesh.model.v1");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CURRENT_WORKFLOW_EDITION, MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION,
        SUPPORTED_WORKFLOW_EDITIONS, capabilities, capability_map,
        is_protocol_generation_compatible, is_workflow_edition_supported,
        missing_required_capabilities, negotiate_workflow_edition, supported_workflow_editions,
    };

    fn offer(editions: &[&str]) -> Vec<String> {
        editions.iter().map(|edition| edition.to_string()).collect()
    }

    #[test]
    fn current_protocol_generation_is_compatible() {
        assert!(is_protocol_generation_compatible(
            PROTOCOL_GENERATION,
            PROTOCOL_GENERATION
        ));
        assert_eq!(MIN_SUPPORTED_PROTOCOL_GENERATION, PROTOCOL_GENERATION);
    }

    #[test]
    fn unknown_protocol_generation_is_not_compatible() {
        assert!(!is_protocol_generation_compatible(
            "rlmesh.protocol.v2",
            PROTOCOL_GENERATION
        ));
        assert!(!is_protocol_generation_compatible(
            PROTOCOL_GENERATION,
            "rlmesh.protocol.v2"
        ));
        assert!(!is_protocol_generation_compatible("", PROTOCOL_GENERATION));
        assert!(!is_protocol_generation_compatible(
            "0.1.0",
            PROTOCOL_GENERATION
        ));
    }

    #[test]
    fn current_workflow_edition_is_supported() {
        assert!(is_workflow_edition_supported(CURRENT_WORKFLOW_EDITION));
        assert_eq!(SUPPORTED_WORKFLOW_EDITIONS, &[CURRENT_WORKFLOW_EDITION]);
        assert_eq!(
            supported_workflow_editions(),
            vec![CURRENT_WORKFLOW_EDITION.to_string()]
        );
    }

    #[test]
    fn unknown_workflow_edition_is_not_supported() {
        assert!(!is_workflow_edition_supported(""));
        assert!(!is_workflow_edition_supported("2026"));
        assert!(!is_workflow_edition_supported("2026.11"));
        assert!(!is_workflow_edition_supported("2027.01"));
    }

    #[test]
    fn negotiation_selects_mutual_edition() {
        assert_eq!(
            negotiate_workflow_edition(&offer(&[CURRENT_WORKFLOW_EDITION])),
            Some(CURRENT_WORKFLOW_EDITION)
        );
        assert_eq!(
            negotiate_workflow_edition(&offer(&["2025.01", CURRENT_WORKFLOW_EDITION, "2031.12"])),
            Some(CURRENT_WORKFLOW_EDITION)
        );
    }

    #[test]
    fn negotiation_trims_offered_editions() {
        assert_eq!(
            negotiate_workflow_edition(&offer(&[" 2026.06 "])),
            Some(CURRENT_WORKFLOW_EDITION)
        );
    }

    #[test]
    fn negotiation_rejects_unknown_or_empty_offers() {
        assert_eq!(negotiate_workflow_edition(&[]), None);
        assert_eq!(negotiate_workflow_edition(&offer(&[""])), None);
        assert_eq!(negotiate_workflow_edition(&offer(&["2026"])), None);
        assert_eq!(negotiate_workflow_edition(&offer(&["next"])), None);
        assert_eq!(
            negotiate_workflow_edition(&offer(&["2026.11", "2027.01"])),
            None
        );
    }

    #[test]
    fn capability_helpers_report_missing_required_features() {
        let offered = capability_map(&[capabilities::ENV_SERVICE_V1, capabilities::SPACES_CORE_V1]);

        assert_eq!(
            missing_required_capabilities(&[capabilities::ENV_SERVICE_V1], &offered),
            Vec::<&str>::new()
        );
        assert_eq!(
            missing_required_capabilities(&[capabilities::MODEL_SERVICE_V1], &offered),
            vec![capabilities::MODEL_SERVICE_V1]
        );
    }
}
