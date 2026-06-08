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
pub const CURRENT_WORKFLOW_EDITION: &str = "2026";

/// Workflow editions understood by this crate.
pub const SUPPORTED_WORKFLOW_EDITIONS: &[&str] = &[CURRENT_WORKFLOW_EDITION];

/// Stable capability names exchanged during handshake.
pub mod capabilities {
    /// Core environment handshake and Join stream.
    pub const ENV_SERVICE_V1: &str = "rlmesh.env.service.v1";

    /// Core model handshake and Join stream.
    pub const MODEL_SERVICE_V1: &str = "rlmesh.model.service.v1";

    /// Core RLMesh space specifications and values.
    pub const SPACES_CORE_V1: &str = "rlmesh.spaces.core.v1";
}

/// Return whether a client protocol generation can speak to a server protocol generation.
pub fn is_protocol_generation_compatible(client: &str, server: &str) -> bool {
    client.trim() == PROTOCOL_GENERATION && server.trim() == PROTOCOL_GENERATION
}

/// Return whether a workflow edition is supported.
pub fn is_workflow_edition_supported(edition: &str) -> bool {
    SUPPORTED_WORKFLOW_EDITIONS.contains(&edition.trim())
}

/// Return supported workflow editions as owned strings for protobuf responses.
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
        missing_required_capabilities, supported_workflow_editions,
    };

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
        assert!(!is_workflow_edition_supported("2027"));
        assert!(!is_workflow_edition_supported(""));
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
