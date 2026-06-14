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

/// The outcome of evaluating a client handshake offer against this build.
///
/// Produced by [`evaluate_handshake`], the single source of truth for the
/// session-compatibility decision shared by the env and model servers — so a
/// careless edit cannot flip interop on one service only.
#[derive(Debug, Clone, Copy)]
pub struct HandshakeCompatibility {
    /// Whether the client's protocol generation can speak to this server.
    pub protocol_compatible: bool,
    /// The negotiated workflow edition, or `None` if there is no mutual edition.
    pub selected_edition: Option<&'static str>,
}

impl HandshakeCompatibility {
    /// Whether the session may proceed: protocol compatible and a mutual edition.
    pub fn is_compatible(&self) -> bool {
        self.protocol_compatible && self.selected_edition.is_some()
    }
}

/// Evaluate a client's handshake offer — its protocol generation and the
/// workflow editions it can operate under — against this build's support.
pub fn evaluate_handshake(
    client_protocol_generation: &str,
    offered_workflow_editions: &[String],
) -> HandshakeCompatibility {
    HandshakeCompatibility {
        protocol_compatible: is_protocol_generation_compatible(
            client_protocol_generation,
            PROTOCOL_GENERATION,
        ),
        selected_edition: negotiate_workflow_edition(offered_workflow_editions),
    }
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
        SUPPORTED_WORKFLOW_EDITIONS, evaluate_handshake, is_protocol_generation_compatible,
        negotiate_workflow_edition, supported_workflow_editions,
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
        assert_eq!(SUPPORTED_WORKFLOW_EDITIONS, &[CURRENT_WORKFLOW_EDITION]);
        assert_eq!(
            supported_workflow_editions(),
            vec![CURRENT_WORKFLOW_EDITION.to_string()]
        );
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
    fn evaluate_handshake_decides_session_compatibility() {
        let current = evaluate_handshake(PROTOCOL_GENERATION, &offer(&[CURRENT_WORKFLOW_EDITION]));
        assert!(current.protocol_compatible);
        assert_eq!(current.selected_edition, Some(CURRENT_WORKFLOW_EDITION));
        assert!(current.is_compatible());

        // Protocol matches but the client predates edition negotiation.
        let no_editions = evaluate_handshake(PROTOCOL_GENERATION, &[]);
        assert!(no_editions.protocol_compatible);
        assert_eq!(no_editions.selected_edition, None);
        assert!(!no_editions.is_compatible());

        // Protocol matches but there is no mutual edition.
        assert!(!evaluate_handshake(PROTOCOL_GENERATION, &offer(&["2099.01"])).is_compatible());

        // A protocol mismatch is never compatible, even with a valid edition.
        let bad_protocol =
            evaluate_handshake("rlmesh.protocol.v2", &offer(&[CURRENT_WORKFLOW_EDITION]));
        assert!(!bad_protocol.protocol_compatible);
        assert!(!bad_protocol.is_compatible());
    }
}
