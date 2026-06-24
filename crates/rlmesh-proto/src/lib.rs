//! Generated RLMesh protobuf bindings and protocol-level constants.

use std::collections::HashMap;
use std::sync::RwLock;

/// Identity of the frozen wire substrate: the `core` handshake plus the `spaces`
/// value transport the runtime relays between env and model. NOT the package
/// version, NOT the product semver, and NOT a per-service version — it names the
/// one shared byte contract every component is built against, decoupled from all
/// of them.
///
/// Compatibility is plain equality: a peer is compatible iff its token equals
/// this. There is no support window and no range negotiation. The wire grows
/// additively forever under this single token; workflow *behavior* rides on
/// editions and optional *features* on capabilities, so neither forces a bump.
///
/// This is a failsafe, not a version counter: in the normal course it is never
/// bumped. Bumping it to `rlmesh-wire-v2` is a deliberate, public hard pivot — a
/// build that intentionally will not interoperate with `rlmesh-wire-v1` — reserved
/// for a wire break that additive growth, editions, and capabilities genuinely
/// cannot absorb.
pub const PROTOCOL_GENERATION: &str = "rlmesh-wire-v1";

/// Current workflow semantics edition.
///
/// Stable releases use a bare sealed `YYYY.MM` label. Prerelease and local
/// source builds use a suffixed cohort (`YYYY.MM-<semver-prerelease>` or
/// `YYYY.MM-dev.<git-token>`) so moving builds only interoperate with the same
/// cohort unless both sides explicitly advertise a sealed fallback edition.
pub const CURRENT_WORKFLOW_EDITION: &str = env!("RLMESH_CURRENT_WORKFLOW_EDITION");

/// Bare `YYYY.MM` workflow edition base for this build.
pub const WORKFLOW_EDITION_BASE: &str = env!("RLMESH_WORKFLOW_EDITION_BASE");

/// Build cohort used to spell [`CURRENT_WORKFLOW_EDITION`].
pub const BUILD_COHORT: &str = env!("RLMESH_BUILD_COHORT");

/// Source of the build cohort: `release`, `package`, or `git`.
pub const BUILD_SOURCE: &str = env!("RLMESH_BUILD_SOURCE");

/// Workflow editions this crate can operate under. Each edition names an
/// immutable behavioral contract documented in `docs/editions/<base>.md` (the
/// spec file is keyed by the bare `YYYY.MM` base, never the suffixed cohort).
pub const SUPPORTED_WORKFLOW_EDITIONS: &[&str] = &[CURRENT_WORKFLOW_EDITION];

/// Stable capability names exchanged during handshake.
///
/// Capabilities are advisory. A present key means the named optional feature is
/// available; an absent key means it is not. They cover optional features that
/// preserve interaction semantics. A feature that changes meaning belongs in an
/// edition or generation. House rule: when an older peer would mishandle an
/// absent field, the emitter checks a capability before sending it; if absence
/// changes semantics, use an edition.
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
    /// Advisory only; this is not an edition change. The wire messages are
    /// identical: every response still mirrors its `request_id`. A client uses
    /// it to decide whether overlapping multiple predicts on one connection will
    /// actually pipeline (capability present) or serialize behind the handler
    /// (capability absent). See `docs/editions/2026.06.md`.
    pub const MODEL_CONCURRENT_PREDICT_V1: &str = "rlmesh.model.concurrent_predict.v1";
}

/// Whether the given protocol generation is the one this build speaks. Plain
/// equality — there is no support window. Whitespace is trimmed so a padded wire
/// value still matches.
pub fn is_protocol_generation_supported(generation: &str) -> bool {
    generation.trim() == PROTOCOL_GENERATION
}

/// Ordering key for a workflow edition name: `(base, cohort?, suffix)`.
///
/// The name is split at its **first** `-` into a `YYYY.MM` base and an optional
/// cohort suffix:
/// - `base` compares lexicographically — the zero-padded fixed-width `YYYY.MM`
///   makes that chronological, so a newer date always outranks an older one.
/// - `cohort?` is `true` for a suffixed prerelease/dev cohort and `false` for a
///   bare sealed fallback, so an exact matching moving cohort wins over its
///   sealed fallback when both peers support it.
/// - `suffix` is the full cohort as a deterministic third tiebreak; two
///   same-date cohorts are ordered by suffix rather than by iteration order.
///
/// Applies to workflow editions only. Protocol generations are compared by plain
/// equality ([`is_protocol_generation_supported`]), never ordered — there is no
/// generation window to pick a highest from.
pub fn edition_sort_key(edition: &str) -> (&str, bool, &str) {
    match edition.split_once('-') {
        Some((base, suffix)) => (base, true, suffix),
        None => (edition, false, ""),
    }
}

/// Select the workflow edition governing a session from a peer's offer.
///
/// Returns the highest edition both peers support, ordered by
/// [`edition_sort_key`] (newest date, then exact moving cohort over sealed
/// fallback, then a deterministic suffix tiebreak). `None` means there is no
/// mutual edition and the handshake must report `compatible = false`. Only
/// explicitly supported editions are eligible. Unknown editions in the offer are
/// ignored, never accepted on the assumption they are compatible.
pub fn negotiate_workflow_edition(offered: &[String]) -> Option<&'static str> {
    SUPPORTED_WORKFLOW_EDITIONS
        .iter()
        .copied()
        .filter(|edition| offered.iter().any(|offer| offer.trim() == *edition))
        .max_by_key(|edition| edition_sort_key(edition))
}

/// One participant's offer in a three-way (relay) session negotiation: the
/// workflow editions it can operate under and the capabilities it advertises.
/// Built for the env, the model, and the runtime; [`negotiate_session_floor`]
/// reconciles all three. Protocol generation is NOT part of the offer — it is
/// gated by plain equality at each pairwise handshake, so a session that reaches
/// floor negotiation already shares one generation across all three.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionOffer {
    /// Workflow editions this participant can operate under.
    pub editions: Vec<String>,
    /// Capabilities this participant advertises (keys of the handshake map whose
    /// value is `"true"`; an absent key means the feature is unavailable).
    pub capabilities: HashMap<String, String>,
}

impl SessionOffer {
    /// Build an offer from string slices, taking only the advertised (value
    /// `"true"`) capabilities. Whitespace around editions is trimmed.
    pub fn new(editions: &[&str], capabilities: &[&str]) -> Self {
        Self {
            editions: editions.iter().map(|e| e.trim().to_string()).collect(),
            capabilities: capability_map(capabilities),
        }
    }

    /// This build's own (runtime) offer: the supported edition window plus the
    /// named capabilities it can carry through the relay.
    pub fn runtime(capabilities: &[&str]) -> Self {
        Self {
            editions: supported_workflow_editions(),
            capabilities: capability_map(capabilities),
        }
    }
}

/// The reconciled three-way session floor produced by [`negotiate_session_floor`].
///
/// A session is bound to these values: the runtime decode-rebuilds env<->model
/// envelopes (prost drops unknown fields), so it can only faithfully carry the
/// shape that env AND model AND runtime all understand. The floor is therefore
/// the highest mutual edition and the capability intersection across all three.
/// Protocol generation is not a floor axis — it is gated by equality at each
/// pairwise handshake, so all three already agree on it here. See
/// versioning-governance §7.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionFloor {
    /// Highest workflow edition all three participants support.
    pub selected_workflow_edition: String,
    /// Capabilities present in all three offers (the intersection).
    pub active_capabilities: HashMap<String, String>,
}

/// Reconcile a three-way (relay) session to its floor over the env, model and
/// runtime offers.
///
/// Returns the highest mutual workflow edition (max over the 3-way intersection,
/// ranked by [`edition_sort_key`]) and the capability intersection across all
/// three. The runtime is the binding authority because it re-frames traffic, so
/// the floor — not any pairwise upper bound — is what a session may use.
/// Protocol generation is not reconciled here: it is gated by equality at each
/// pairwise handshake, so all three already share it by the time this runs.
///
/// Returns `None` when no edition is common to all three; the caller must then
/// fail the session before any Join stream opens. Whitespace around offered
/// editions is trimmed; empty strings never match.
pub fn negotiate_session_floor(
    env: &SessionOffer,
    model: &SessionOffer,
    runtime: &SessionOffer,
) -> Option<SessionFloor> {
    let selected_workflow_edition = highest_mutual(
        &env.editions,
        &model.editions,
        &runtime.editions,
        edition_sort_key,
    )?;

    // Capability intersection: a capability is active only when all three
    // advertise it (value "true"). The runtime must carry any field the feature
    // needs, so env<->model agreement alone is insufficient.
    let active_capabilities = env
        .capabilities
        .iter()
        .filter(|(name, value)| {
            value.as_str() == "true"
                && has_capability(&model.capabilities, name)
                && has_capability(&runtime.capabilities, name)
        })
        .map(|(name, _)| (name.clone(), "true".to_string()))
        .collect();

    Some(SessionFloor {
        selected_workflow_edition,
        active_capabilities,
    })
}

/// Highest edition present in all three sets (after trimming), ranked by `key`,
/// or `None` if the three-way intersection is empty. Empty strings never match.
fn highest_mutual<'a, K: Ord>(
    a: &'a [String],
    b: &[String],
    c: &[String],
    key: impl Fn(&'a str) -> K,
) -> Option<String> {
    a.iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .filter(|value| b.iter().any(|other| other.trim() == *value))
        .filter(|value| c.iter().any(|other| other.trim() == *value))
        .max_by_key(|value| key(value))
        .map(|value| value.to_string())
}

/// The outcome of evaluating a client handshake offer against this build.
///
/// Produced by [`evaluate_handshake`]. Env and model servers use this same
/// result, so compatibility cannot drift between services.
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

/// Evaluate a client's handshake offer against this build's supported protocol
/// generation and workflow editions.
pub fn evaluate_handshake(
    client_protocol_generation: &str,
    offered_workflow_editions: &[String],
) -> HandshakeCompatibility {
    HandshakeCompatibility {
        protocol_compatible: is_protocol_generation_supported(client_protocol_generation),
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

/// Whether a peer's handshake capability map advertises the named capability.
pub fn has_capability(map: &HashMap<String, String>, name: &str) -> bool {
    map.get(name).is_some_and(|value| value == "true")
}

pub mod core {
    pub mod v1 {
        tonic::include_proto!("rlmesh.core.v1");
    }
}

/// Advisory runtime identity supplied by a non-Rust host (e.g. the Python SDK)
/// to enrich the handshake [`PeerInfo`](core::v1::PeerInfo).
///
/// Every field is optional and best-effort. When set process-wide via
/// [`set_peer_info_override`], [`peer_info`] merges these values over the
/// Rust-detected defaults: a non-empty override field wins, an empty/absent one
/// falls back to the Rust-detected value (`os`/`arch`/`package_version`). The
/// `component` passed to [`peer_info`] is always honored; an override
/// `component` is ignored so each call site keeps naming itself.
///
/// This is purely additive diagnostics: PeerInfo never gates compatibility.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerInfoOverride {
    /// Implementation language, e.g. `"python"`. Empty leaves the Rust default.
    pub language: String,
    /// Language runtime version, e.g. `"3.11.4"`.
    pub language_version: String,
    /// Package/build version of the host SDK. Empty falls back to this crate's.
    pub package_version: String,
    /// Operating system, e.g. `"linux"`, `"macos"`. Empty falls back to the
    /// Rust-detected [`std::env::consts::OS`].
    pub os: String,
    /// OS version/release.
    pub os_version: String,
    /// CPU architecture, e.g. `"x86_64"`. Empty falls back to the Rust-detected
    /// [`std::env::consts::ARCH`].
    pub arch: String,
    /// High-value framework versions for debugging (e.g. `{"numpy":"1.26.4"}`).
    pub framework_versions: HashMap<String, String>,
    /// Additional advisory key/value diagnostics.
    pub extra: HashMap<String, String>,
}

/// Process-wide host identity override consulted by [`peer_info`].
///
/// `None` (the default) means no override: pure-Rust peers handshake exactly as
/// before. A Python-hosted process sets this once at import; the value applies
/// to every handshake the process performs (it is the only host).
static PEER_INFO_OVERRIDE: RwLock<Option<PeerInfoOverride>> = RwLock::new(None);

/// Install (or replace) the process-wide [`PeerInfoOverride`] consulted by
/// [`peer_info`]. Intended for non-Rust hosts (the Python SDK) to report their
/// real runtime. Idempotent and thread-safe; passing the value again overwrites.
pub fn set_peer_info_override(info: PeerInfoOverride) {
    if let Ok(mut guard) = PEER_INFO_OVERRIDE.write() {
        *guard = Some(info);
    }
}

/// Build advisory [`PeerInfo`](core::v1::PeerInfo) diagnostics for a handshake.
///
/// `component` names the emitting participant (e.g. `"rlmesh-runtime"`,
/// `"rlmesh-env"`, `"rlmesh-model"`). The build version, language, OS and arch
/// default to this crate's compile environment (`language="rust"`, empty
/// `language_version`/`os_version`/`framework_versions`).
///
/// When a process-wide [`PeerInfoOverride`] has been installed via
/// [`set_peer_info_override`] (e.g. by the Python SDK), its non-empty fields win
/// over the Rust defaults, falling back to the Rust-detected
/// `os`/`arch`/`package_version` for any empty override field. The `component`
/// argument is always preserved so each call site keeps naming itself. PeerInfo
/// is advisory only and never gates compatibility.
pub fn peer_info(component: &str) -> core::v1::PeerInfo {
    let mut extra = HashMap::new();
    extra.insert(
        "rlmesh.workflow.base".to_string(),
        WORKFLOW_EDITION_BASE.to_string(),
    );
    extra.insert(
        "rlmesh.workflow.edition".to_string(),
        CURRENT_WORKFLOW_EDITION.to_string(),
    );
    extra.insert("rlmesh.build.cohort".to_string(), BUILD_COHORT.to_string());
    extra.insert("rlmesh.build.source".to_string(), BUILD_SOURCE.to_string());

    let mut info = core::v1::PeerInfo {
        component: component.to_string(),
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        language: "rust".to_string(),
        language_version: String::new(),
        os: std::env::consts::OS.to_string(),
        os_version: String::new(),
        arch: std::env::consts::ARCH.to_string(),
        framework_versions: HashMap::new(),
        extra,
    };

    if let Ok(guard) = PEER_INFO_OVERRIDE.read()
        && let Some(over) = guard.as_ref()
    {
        // Python (or other host) values win when present; empty fields keep the
        // Rust-detected fallback. `component` is never overridden.
        if !over.language.is_empty() {
            info.language = over.language.clone();
        }
        if !over.language_version.is_empty() {
            info.language_version = over.language_version.clone();
        }
        if !over.package_version.is_empty() {
            info.package_version = over.package_version.clone();
        }
        if !over.os.is_empty() {
            info.os = over.os.clone();
        }
        if !over.os_version.is_empty() {
            info.os_version = over.os_version.clone();
        }
        if !over.arch.is_empty() {
            info.arch = over.arch.clone();
        }
        if !over.framework_versions.is_empty() {
            info.framework_versions = over.framework_versions.clone();
        }
        if !over.extra.is_empty() {
            info.extra.extend(over.extra.clone());
        }
    }

    info
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
        CURRENT_WORKFLOW_EDITION, PROTOCOL_GENERATION, SUPPORTED_WORKFLOW_EDITIONS,
        evaluate_handshake, is_protocol_generation_supported, negotiate_workflow_edition,
        supported_workflow_editions,
    };

    fn offer(editions: &[&str]) -> Vec<String> {
        editions.iter().map(|edition| edition.to_string()).collect()
    }

    #[test]
    fn peer_info_default_then_override_merges_python_with_rust_fallback() {
        use super::{PeerInfoOverride, peer_info, set_peer_info_override};
        use std::collections::HashMap;

        // No override installed yet: a pure-Rust peer reports the Rust defaults.
        let rust_info = peer_info("rlmesh-env");
        assert_eq!(rust_info.component, "rlmesh-env");
        assert_eq!(rust_info.language, "rust");
        assert!(rust_info.language_version.is_empty());
        assert!(rust_info.framework_versions.is_empty());
        let detected_os = rust_info.os.clone();
        let detected_arch = rust_info.arch.clone();
        let detected_pkg = rust_info.package_version.clone();

        // Install a Python-style override with `os`/`package_version` left empty
        // so the Rust-detected fallbacks fill them.
        let mut frameworks = HashMap::new();
        frameworks.insert("numpy".to_string(), "1.26.4".to_string());
        set_peer_info_override(PeerInfoOverride {
            language: "python".to_string(),
            language_version: "3.11.4".to_string(),
            package_version: String::new(),
            os: String::new(),
            os_version: "ubuntu-22.04".to_string(),
            arch: "aarch64".to_string(),
            framework_versions: frameworks,
            extra: HashMap::new(),
        });

        let py_info = peer_info("rlmesh-env");
        // component still names this call site; not taken from the override.
        assert_eq!(py_info.component, "rlmesh-env");
        // Python values win.
        assert_eq!(py_info.language, "python");
        assert_eq!(py_info.language_version, "3.11.4");
        assert_eq!(py_info.os_version, "ubuntu-22.04");
        assert_eq!(py_info.arch, "aarch64");
        assert_eq!(
            py_info.framework_versions.get("numpy").map(String::as_str),
            Some("1.26.4")
        );
        // Empty override fields fall back to the Rust-detected values.
        assert_eq!(py_info.os, detected_os);
        assert_eq!(py_info.package_version, detected_pkg);
        assert_eq!(
            py_info
                .extra
                .get("rlmesh.workflow.edition")
                .map(String::as_str),
            Some(CURRENT_WORKFLOW_EDITION)
        );
        // `arch` was overridden, so it differs from the detected value here.
        let _ = detected_arch;
    }

    #[test]
    fn has_capability_reads_advertised_features() {
        use super::{capabilities, capability_map, has_capability};
        let map = capability_map(&[capabilities::ENV_SERVICE_V1]);
        assert!(has_capability(&map, capabilities::ENV_SERVICE_V1));
        assert!(!has_capability(
            &map,
            capabilities::MODEL_CONCURRENT_PREDICT_V1
        ));
    }

    #[test]
    fn protocol_generation_is_plain_equality() {
        // The only generation check is equality with this build's generation —
        // there is no support window. Whitespace is trimmed; anything else is a
        // hard mismatch (a deliberate major break).
        assert!(is_protocol_generation_supported(PROTOCOL_GENERATION));
        assert!(is_protocol_generation_supported(&format!(
            " {PROTOCOL_GENERATION} "
        )));
        assert!(!is_protocol_generation_supported("rlmesh-wire-v2"));
        assert!(!is_protocol_generation_supported(""));
        assert!(!is_protocol_generation_supported("0.1.0"));
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
        let padded = format!(" {CURRENT_WORKFLOW_EDITION} ");
        assert_eq!(
            negotiate_workflow_edition(&[padded]),
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
            evaluate_handshake("rlmesh-wire-v2", &offer(&[CURRENT_WORKFLOW_EDITION]));
        assert!(!bad_protocol.protocol_compatible);
        assert!(!bad_protocol.is_compatible());
    }

    #[test]
    fn session_floor_picks_highest_mutual_edition() {
        use super::{SessionOffer, negotiate_session_floor};
        // All three share e1 and e2 → e2 wins. Generation is not a floor axis; it
        // is gated by equality at the handshake before this runs.
        let env = SessionOffer::new(&["2026.01", "2026.06"], &[]);
        let model = SessionOffer::new(&["2026.06", "2026.01"], &[]);
        let runtime = SessionOffer::new(&["2026.06"], &[]);

        let floor = negotiate_session_floor(&env, &model, &runtime).expect("a mutual floor");
        assert_eq!(floor.selected_workflow_edition, "2026.06");
        assert!(floor.active_capabilities.is_empty());
    }

    #[test]
    fn session_floor_intersects_capabilities_across_all_three() {
        use super::{SessionOffer, has_capability, negotiate_session_floor};
        // shared: present in all three. env_model_only: env+model but not runtime.
        // runtime_only: runtime alone. None but "shared" survive the intersection.
        let env = SessionOffer::new(&["e1"], &["shared", "env_model_only", "env_only"]);
        let model = SessionOffer::new(&["e1"], &["shared", "env_model_only"]);
        let runtime = SessionOffer::new(&["e1"], &["shared", "runtime_only"]);

        let floor = negotiate_session_floor(&env, &model, &runtime).expect("a mutual floor");
        assert_eq!(floor.active_capabilities.len(), 1);
        assert!(has_capability(&floor.active_capabilities, "shared"));
        assert!(!has_capability(
            &floor.active_capabilities,
            "env_model_only"
        ));
        assert!(!has_capability(&floor.active_capabilities, "runtime_only"));
        assert!(!has_capability(&floor.active_capabilities, "env_only"));
    }

    #[test]
    fn session_floor_is_none_when_no_three_way_mutual_edition() {
        use super::{SessionOffer, negotiate_session_floor};
        // The pivotal case: env<->runtime share 2026.01 and model<->runtime share
        // 2026.06, but the THREE share no edition → None.
        let env = SessionOffer::new(&["2026.01"], &[]);
        let model = SessionOffer::new(&["2026.06"], &[]);
        let runtime = SessionOffer::new(&["2026.01", "2026.06"], &[]);
        assert!(negotiate_session_floor(&env, &model, &runtime).is_none());
    }

    #[test]
    fn session_floor_trims_and_ignores_empty_offers() {
        use super::{SessionOffer, negotiate_session_floor};
        // Whitespace is trimmed so a padded edition still matches; empty strings
        // never match (so a participant offering only "" has no mutual value).
        let env = SessionOffer::new(&[" 2026.06 "], &[]);
        let model = SessionOffer::new(&["2026.06"], &[]);
        let runtime = SessionOffer::new(&["2026.06", ""], &[]);
        let floor = negotiate_session_floor(&env, &model, &runtime).expect("trimmed match");
        assert_eq!(floor.selected_workflow_edition, "2026.06");

        let empty_edition = SessionOffer::new(&[""], &[]);
        assert!(negotiate_session_floor(&empty_edition, &model, &runtime).is_none());
    }

    #[test]
    fn session_floor_for_single_edition_build() {
        use super::{
            CURRENT_WORKFLOW_EDITION, SessionOffer, capabilities, has_capability,
            negotiate_session_floor,
        };
        // The behavior this build actually ships: one edition. The floor is
        // trivially that edition, and the capability intersection is exactly the
        // capabilities all three carry.
        let env = SessionOffer::new(
            &[CURRENT_WORKFLOW_EDITION],
            &[capabilities::SPACES_CORE_V1, capabilities::ENV_SERVICE_V1],
        );
        let model = SessionOffer::new(
            &[CURRENT_WORKFLOW_EDITION],
            &[capabilities::SPACES_CORE_V1, capabilities::MODEL_SERVICE_V1],
        );
        let runtime = SessionOffer::runtime(&[capabilities::SPACES_CORE_V1]);

        let floor = negotiate_session_floor(&env, &model, &runtime).expect("single-edition floor");
        assert_eq!(floor.selected_workflow_edition, CURRENT_WORKFLOW_EDITION);
        // Only the capability all three carry survives.
        assert!(has_capability(
            &floor.active_capabilities,
            capabilities::SPACES_CORE_V1
        ));
        assert_eq!(floor.active_capabilities.len(), 1);
    }

    #[test]
    fn session_floor_capability_requires_value_true_in_all_three() {
        use super::{SessionOffer, has_capability, negotiate_session_floor};
        // A capability advertised with value != "true" by any participant is not
        // active (has_capability gates on "true"); mirror that in the floor.
        let mut env = SessionOffer::new(&["e1"], &["cap"]);
        let model = SessionOffer::new(&["e1"], &["cap"]);
        let runtime = SessionOffer::new(&["e1"], &["cap"]);
        // Downgrade env's advertisement to a non-"true" value.
        env.capabilities
            .insert("cap".to_string(), "maybe".to_string());

        let floor = negotiate_session_floor(&env, &model, &runtime).expect("floor");
        assert!(!has_capability(&floor.active_capabilities, "cap"));
    }

    #[test]
    fn edition_ordering_prefers_exact_cohort_then_newer_date() {
        use super::edition_sort_key;

        // Exact moving cohorts beat their own sealed fallback. This lets two
        // matching prerelease/dev peers use the newest cohort while still allowing
        // fallback to the sealed edition when the moving cohorts differ.
        assert!(edition_sort_key("2026.06-0.1.0-rc.1") > edition_sort_key("2026.06"));

        // newer-date-wins: a newer date outranks an older one regardless of
        // cohort status.
        assert!(edition_sort_key("2026.09-0.2.0-beta.1") > edition_sort_key("2026.06"));
        assert!(edition_sort_key("2026.09") > edition_sort_key("2026.06-0.1.0-rc.1"));

        // deterministic suffix tiebreak: two same-date cohorts order by
        // their full suffix, never by iteration order, so two honest builds
        // never disagree on the winner.
        assert!(edition_sort_key("2026.06-0.1.0-rc.2") > edition_sort_key("2026.06-0.1.0-rc.1"));

        // negotiate_workflow_edition applies the same key: offered against a
        // hypothetical multi-edition supported set, the highest by key wins. With
        // the single supported edition this build ships, the current edition is
        // selected when offered alongside older/newer noise.
        assert_eq!(
            negotiate_workflow_edition(&offer(&["2025.01", CURRENT_WORKFLOW_EDITION, "2099.12"])),
            Some(CURRENT_WORKFLOW_EDITION)
        );
    }

    #[test]
    fn session_floor_prefers_exact_edition_cohort_over_sealed_fallback() {
        use super::{SessionOffer, negotiate_session_floor};
        // The floor uses edition_sort_key: when all three offer both the exact
        // moving cohort and its sealed fallback, the exact cohort is selected.
        let editions = &["2026.06", "2026.06-0.1.0-rc.1"];
        let env = SessionOffer::new(editions, &[]);
        let model = SessionOffer::new(editions, &[]);
        let runtime = SessionOffer::new(editions, &[]);
        let floor = negotiate_session_floor(&env, &model, &runtime).expect("a mutual floor");
        assert_eq!(floor.selected_workflow_edition, "2026.06-0.1.0-rc.1");
    }
}
