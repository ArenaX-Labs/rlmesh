//! Internal driver data types, workflow editions, and build identity.
#![deny(rustdoc::broken_intra_doc_links)]

use std::collections::HashMap;
use std::sync::RwLock;

/// Identity of the realtime session wire used by this build.
pub const PROTOCOL_GENERATION: &str = "rlmesh-realtime-v1";

/// The workflow edition this build runs at when nothing declares one: a bare
/// `YYYY.MM`, the same on every build of a release line.
pub const CURRENT_WORKFLOW_EDITION: &str = "2026.10";

/// How this build was made: `release`, `package`, or `git`.
pub const BUILD_SOURCE: &str = env!("RLMESH_BUILD_SOURCE");

/// The short git commit a source build was stamped from; empty otherwise.
pub const BUILD_GIT: &str = env!("RLMESH_BUILD_GIT");

/// A workflow edition this build knows: one row of behavioural defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Edition {
    /// The `2026.10` edition.
    E2026_10,
}

impl Edition {
    /// Every edition this build knows, oldest first.
    pub const ALL: &'static [Edition] = &[Edition::E2026_10];

    /// The edition's `YYYY.MM` name.
    pub const fn name(&self) -> &'static str {
        match self {
            Edition::E2026_10 => "2026.10",
        }
    }

    /// The edition this build runs at when nothing declares one
    /// ([`CURRENT_WORKFLOW_EDITION`]).
    pub fn current() -> Edition {
        Edition::parse(CURRENT_WORKFLOW_EDITION).expect("CURRENT_WORKFLOW_EDITION names an arm")
    }

    /// Resolve an edition name to the edition it names.
    ///
    /// # Errors
    /// [`UnknownEdition`] when this build does not know the name, typically an
    /// edition from a newer release.
    pub fn parse(edition: &str) -> Result<Edition, UnknownEdition> {
        let name = edition.trim();
        Edition::ALL
            .iter()
            .copied()
            .find(|edition| edition.name() == name)
            .ok_or_else(|| UnknownEdition(name.to_string()))
    }
}

impl std::fmt::Display for Edition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// An edition name this build does not know.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownEdition(String);

impl UnknownEdition {
    /// The rejected name, trimmed.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for UnknownEdition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let known = Edition::ALL
            .iter()
            .map(Edition::name)
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "workflow edition {:?} is unknown to this rlmesh build, which knows {known}; \
             upgrade the rlmesh runtime to a release that knows {:?}, or rebuild the image on \
             a release whose edition this runtime knows",
            self.0, self.0,
        )
    }
}

impl std::error::Error for UnknownEdition {}

/// The defaults one [`Edition`] promises: the values a participant would
/// otherwise hardcode, so changing one is a new edition rather than an edit to
/// a code path an older edition depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditionDefaults {
    /// Per-episode step bound applied when the session sets no explicit
    /// `max_episode_steps` and the runtime owns resets, so a broken termination
    /// condition surfaces as a truncation instead of hanging the run.
    pub default_max_episode_steps: i64,

    /// The single reserved `ResetRequest.options` key this edition defines.
    pub trial_index_option_key: &'static str,

    /// The autoreset modes under which the RUNTIME owns lane restarts (it issues
    /// the `Reset`s), as opposed to the env rolling its own lanes.
    pub driver_owned_reset_modes: &'static [core::v1::AutoresetMode],

    /// The `final_info` keys this edition reads an episode's task outcome from,
    /// in priority order: the first key holding a bool, integer, or number
    /// decides, a numeric value coercing by truthiness (`1`/`1.0` → true); a key
    /// holding any other kind is skipped.
    pub success_info_keys: &'static [&'static str],

    /// The reserved info-map key a served env reports value-conformance
    /// warnings under.
    pub conformance_warning_info_key: &'static str,
}

static DEFAULTS_2026_10: EditionDefaults = EditionDefaults {
    default_max_episode_steps: 100_000,
    trial_index_option_key: "trial_index",
    driver_owned_reset_modes: &[
        core::v1::AutoresetMode::Disabled,
        core::v1::AutoresetMode::Unspecified,
    ],
    success_info_keys: &["is_success", "success", "task_success"],
    conformance_warning_info_key: "rlmesh.conformance.warning",
};

/// The defaults row of `edition`.
pub const fn defaults(edition: Edition) -> &'static EditionDefaults {
    match edition {
        Edition::E2026_10 => &DEFAULTS_2026_10,
    }
}

/// Stable capability names exchanged during handshake.
pub mod capabilities {
    /// A served model endpoint processes Join-stream requests concurrently
    /// (pipelined predict): responses arrive in completion order rather than
    /// strict arrival order, while per-route lifecycle ordering is preserved.
    pub const MODEL_CONCURRENT_PREDICT_V1: &str = "rlmesh.model.concurrent_predict.v1";

    /// A served model endpoint understands observation history: it answers
    /// `ResolveAdapterResponse.history` when the runtime offers
    /// `delivers_history`, ingests `PredictRequest.history` rows, and holds the
    /// producer to consecutive `step`s.
    pub const MODEL_OBSERVATION_HISTORY_V1: &str = "rlmesh.model.observation_history.v1";

    /// A served env endpoint steps and resets lanes individually: a `Reset` or
    /// `Step` naming `env_indices` is honored (and answered partial-width)
    /// instead of rejected with `UNSUPPORTED`, and lane-scoped requests on one
    /// Join stream may be processed concurrently, answered in completion order.
    pub const ENV_SUBSET_STEP: &str = "subset_step";
}

/// Return a handshake capability map for the given capability names.
///
/// # Value grammar
///
/// A handshake capability map is keyed by capability name; the value is the
/// literal string `"true"` and nothing else is defined by this protocol
/// generation. Every RLMesh emitter goes through this function, so every value
/// RLMesh puts on the wire is `"true"`, and [`has_capability`] is the matching
/// reader: a key whose value is anything else — `"1"`, `"yes"`, `"TRUE"`, or
/// the empty string — reads as NOT advertised, exactly like an absent key. A
/// third-party peer that wants a capability honored must therefore send
/// `"true"` verbatim.
///
/// Capabilities are advisory, so this strictness is safe in both directions:
/// the worst case of an unrecognized value is the feature staying off.
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
        include!(concat!(env!("OUT_DIR"), "/rlmesh.core.v1.rs"));
    }
}

/// Advisory runtime identity supplied by a non-Rust host (e.g. the Python SDK)
/// to enrich the handshake [`PeerInfo`](core::v1::PeerInfo).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PeerInfoOverride {
    /// Implementation language, e.g. `"python"`.
    pub language: String,
    /// Language runtime version, e.g. `"3.11.4"`.
    pub language_version: String,
    /// Package/build version of the host SDK.
    pub package_version: String,
    /// Operating system, e.g. `"linux"`, `"macos"`.
    pub os: String,
    /// OS version/release.
    pub os_version: String,
    /// CPU architecture, e.g. `"x86_64"`.
    pub arch: String,
    /// High-value framework versions for debugging (e.g. `{"numpy":"1.26.4"}`).
    pub framework_versions: HashMap<String, String>,
    /// Additional advisory key/value diagnostics.
    pub extra: HashMap<String, String>,
}

static PEER_INFO_OVERRIDE: RwLock<Option<PeerInfoOverride>> = RwLock::new(None);

/// Install (or replace) the process-wide [`PeerInfoOverride`] consulted by
/// [`peer_info`].
pub fn set_peer_info_override(info: PeerInfoOverride) {
    if let Ok(mut guard) = PEER_INFO_OVERRIDE.write() {
        *guard = Some(info);
    }
}

/// The installed process-wide [`PeerInfoOverride`], if any.
pub fn peer_info_override() -> Option<PeerInfoOverride> {
    PEER_INFO_OVERRIDE
        .read()
        .ok()
        .and_then(|guard| guard.clone())
}

/// Edit the process-wide [`PeerInfoOverride`] in place, starting from an empty
/// one when none is installed, so a host can add one field (e.g. an
/// [`extra`](PeerInfoOverride::extra) key) without clobbering the rest.
pub fn update_peer_info_override(edit: impl FnOnce(&mut PeerInfoOverride)) {
    if let Ok(mut guard) = PEER_INFO_OVERRIDE.write() {
        edit(guard.get_or_insert_with(PeerInfoOverride::default));
    }
}

/// Build advisory [`PeerInfo`](core::v1::PeerInfo) diagnostics for a handshake.
pub fn peer_info(component: &str) -> core::v1::PeerInfo {
    let mut extra = HashMap::new();
    extra.insert(
        "rlmesh.workflow.edition".to_string(),
        CURRENT_WORKFLOW_EDITION.to_string(),
    );
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
        include!(concat!(env!("OUT_DIR"), "/rlmesh.env.v1.rs"));
    }
}

pub mod spaces {
    pub mod v1 {
        include!(concat!(env!("OUT_DIR"), "/rlmesh.spaces.v1.rs"));
    }
}

pub mod model {
    pub mod v1 {
        include!(concat!(env!("OUT_DIR"), "/rlmesh.model.v1.rs"));
    }
}

/// The endpoint-local split of `JoinResponse.endpoint_total_ns` (ns), plus the
/// pre-handler wait and slot depth a pipelining model endpoint reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndpointPhases {
    /// Decoding the request off the wire.
    pub decode_ns: u64,
    /// The env or model implementation's own work, user code included.
    pub user_ns: u64,
    /// Encoding the response onto the wire.
    pub encode_ns: u64,
    /// Wait before the handler ran.
    pub queue_ns: u64,
    /// Adapter work inside `user_ns` (observation assembly + action apply) —
    /// a sub-span, not additive: the model's own forward is `user_ns -
    /// adapter_ns`.
    pub adapter_ns: u64,
    /// Episodes whose frame-stack windows the endpoint's adapter engine held
    /// when the response was stamped, across all routes.
    pub held_episodes: Option<u32>,
    /// Bytes those held frame-stack windows occupy.
    pub held_state_bytes: Option<u64>,
    /// Requests holding a concurrency slot when this request was dispatched to
    /// its handler (>= 1): slot occupancy, not parallelism.
    pub in_flight: u32,
    /// Straggler skew across a vector env's lanes for this op — see
    /// [`lane_skew_ns`].
    pub lane_skew_ns: Option<u64>,
}

impl EndpointPhases {
    /// Wire form of a measurement: an unmeasured (zero) phase is left off the
    /// message rather than sent as a `0` a reader cannot tell from silence.
    pub fn reported(ns: u64) -> Option<u64> {
        (ns != 0).then_some(ns)
    }

    /// Build a split from measured phase durations, saturating at `u64::MAX` ns.
    pub fn from_durations(
        decode: std::time::Duration,
        user: std::time::Duration,
        encode: std::time::Duration,
    ) -> Self {
        fn ns(duration: std::time::Duration) -> u64 {
            duration.as_nanos().min(u128::from(u64::MAX)) as u64
        }
        Self {
            decode_ns: ns(decode),
            user_ns: ns(user),
            encode_ns: ns(encode),
            ..Self::default()
        }
    }

    /// Fold an inner layer's own split into this layer's decode/call/encode
    /// measurement.
    pub fn nest(decode_ns: u64, call_ns: u64, encode_ns: u64, inner: Self) -> Self {
        let inner_wire = inner.decode_ns.saturating_add(inner.encode_ns);
        if inner_wire > call_ns {
            return Self {
                decode_ns,
                user_ns: call_ns,
                encode_ns,
                lane_skew_ns: inner.lane_skew_ns,
                ..Self::default()
            };
        }
        Self {
            decode_ns: decode_ns.saturating_add(inner.decode_ns),
            user_ns: call_ns - inner_wire,
            encode_ns: encode_ns.saturating_add(inner.encode_ns),
            lane_skew_ns: inner.lane_skew_ns,
            ..Self::default()
        }
    }

    /// Read the split an env peer stamped; all-zero for a peer that sends none.
    pub fn from_env_response(response: &env::v1::JoinResponse) -> Self {
        Self {
            decode_ns: response.decode_ns.unwrap_or(0),
            user_ns: response.user_ns.unwrap_or(0),
            encode_ns: response.encode_ns.unwrap_or(0),
            queue_ns: response.queue_ns.unwrap_or(0),
            lane_skew_ns: response.lane_skew_ns,
            ..Self::default()
        }
    }

    /// Read the split a model peer stamped; all-zero for a peer that sends none.
    pub fn from_model_response(response: &model::v1::JoinResponse) -> Self {
        Self {
            decode_ns: response.decode_ns.unwrap_or(0),
            user_ns: response.user_ns.unwrap_or(0),
            encode_ns: response.encode_ns.unwrap_or(0),
            queue_ns: response.queue_ns.unwrap_or(0),
            in_flight: response.in_flight.unwrap_or(0),
            adapter_ns: response.adapter_ns.unwrap_or(0),
            held_episodes: response.held_episodes,
            held_state_bytes: response.held_state_bytes,
            ..Self::default()
        }
    }
}

/// Nanoseconds elapsed since `started_at`, saturating instead of wrapping.
pub fn elapsed_ns(started_at: std::time::Instant) -> u64 {
    started_at.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

/// Straggler skew across a vector env's lanes for one op: how much longer the
/// slowest lane took than the median lane, in nanoseconds.
pub fn lane_skew_ns(lane_ns: &mut [u64]) -> u64 {
    if lane_ns.len() < 2 {
        return 0;
    }
    let (_, median, slower) = lane_ns.select_nth_unstable((lane_ns.len() - 1) / 2);
    let median = *median;
    slower
        .iter()
        .copied()
        .max()
        .unwrap_or(median)
        .saturating_sub(median)
}

/// A completing lane's final info from a step's `infos`, `lane` being its
/// position among the `width` lanes the step covered.
pub fn lane_final_info(
    info: Option<&spaces::v1::MetaMap>,
    lane: usize,
    width: usize,
) -> Option<spaces::v1::MetaMap> {
    use spaces::v1::meta_value::Kind;
    let info = info?;
    let Some(final_info) = info.entries.get("final_info") else {
        if width == 1 {
            return Some(info.clone());
        }
        let sliced = vector_info_lane(info, lane, width);
        return (!sliced.entries.is_empty()).then_some(sliced);
    };
    let is_present = match info.entries.get("_final_info") {
        Some(mask) => meta_bool_at(mask, lane).unwrap_or(false),
        None => width == 1,
    };
    if !is_present {
        return None;
    }
    match &final_info.kind {
        Some(Kind::Map(map)) => Some(map.clone()),
        Some(Kind::List(list)) => match &list.items.get(lane)?.kind {
            Some(Kind::Map(map)) => Some(map.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn vector_info_lane(info: &spaces::v1::MetaMap, lane: usize, width: usize) -> spaces::v1::MetaMap {
    use spaces::v1::meta_value::Kind;
    let entries = info
        .entries
        .iter()
        .filter(|(key, _)| {
            !key.strip_prefix('_')
                .is_some_and(|masked| info.entries.contains_key(masked))
        })
        .filter(|(key, _)| {
            info.entries
                .get(&format!("_{key}"))
                .is_none_or(|mask| meta_bool_at(mask, lane) == Some(true))
        })
        .filter_map(|(key, value)| {
            let lane_value = match &value.kind {
                Some(Kind::Map(map)) => spaces::v1::MetaValue {
                    kind: Some(Kind::Map(vector_info_lane(map, lane, width))),
                },
                Some(Kind::List(list)) if list.items.len() == width => list.items[lane].clone(),
                _ => return None,
            };
            Some((key.clone(), lane_value))
        })
        .collect();
    spaces::v1::MetaMap { entries }
}

fn meta_bool_at(value: &spaces::v1::MetaValue, lane: usize) -> Option<bool> {
    use spaces::v1::meta_value::Kind;
    match &value.kind {
        Some(Kind::List(list)) => match list.items.get(lane)?.kind {
            Some(Kind::Bool(flag)) => Some(flag),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{CURRENT_WORKFLOW_EDITION, Edition, defaults};

    #[test]
    fn every_edition_has_a_row_and_parses_by_its_name() {
        for edition in Edition::ALL {
            let row = defaults(*edition);
            assert!(!row.trial_index_option_key.is_empty());
            assert!(!row.conformance_warning_info_key.is_empty());
            assert_eq!(Edition::parse(edition.name()), Ok(*edition));
            assert_eq!(edition.to_string(), edition.name());
        }
        assert_eq!(Edition::ALL.last(), Some(&Edition::current()));
        assert_eq!(
            Edition::parse(&format!("  {CURRENT_WORKFLOW_EDITION}  ")),
            Ok(Edition::current())
        );
    }

    #[test]
    fn an_unknown_edition_is_refused_with_the_known_ones_and_the_remedy() {
        for name in [
            "2099.01",
            "",
            "2026.10-0.2.0-alpha.1",
            "2026.10-dev.1a2b3c4d5e6f",
        ] {
            let error = Edition::parse(name).expect_err("not a known edition");
            assert_eq!(error.name(), name);
            let message = error.to_string();
            assert!(
                message.contains(&format!("{name:?}"))
                    && message.contains(CURRENT_WORKFLOW_EDITION)
                    && message.contains("upgrade the rlmesh runtime")
                    && message.contains("rebuild the image"),
                "{message}"
            );
        }
    }

    #[test]
    fn lane_final_info_reads_each_gymnasium_info_layout() {
        use super::lane_final_info;
        use super::spaces::v1::meta_value::Kind;
        use super::spaces::v1::{MetaList, MetaMap, MetaValue};

        let value = |kind: Kind| MetaValue { kind: Some(kind) };
        let list = |items: Vec<MetaValue>| value(Kind::List(MetaList { items }));
        let map = |entries: Vec<(&str, MetaValue)>| MetaMap {
            entries: entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        };
        let success = |flag: bool| map(vec![("is_success", value(Kind::Bool(flag)))]);

        let scalar = success(false);
        assert_eq!(lane_final_info(Some(&scalar), 0, 1), Some(success(false)));
        assert_eq!(lane_final_info(None, 0, 2), None);

        let explicit = map(vec![
            (
                "final_info",
                list(vec![
                    value(Kind::Map(success(true))),
                    value(Kind::Map(MetaMap::default())),
                ]),
            ),
            (
                "_final_info",
                list(vec![value(Kind::Bool(true)), value(Kind::Bool(false))]),
            ),
        ]);
        assert_eq!(lane_final_info(Some(&explicit), 0, 2), Some(success(true)));
        assert_eq!(lane_final_info(Some(&explicit), 1, 2), None);

        let unmasked = map(vec![(
            "is_success",
            list(vec![value(Kind::Bool(false)), value(Kind::Bool(true))]),
        )]);
        assert_eq!(lane_final_info(Some(&unmasked), 1, 2), Some(success(true)));

        let masked_out = map(vec![
            (
                "is_success",
                list(vec![value(Kind::Bool(true)), value(Kind::Bool(false))]),
            ),
            (
                "_is_success",
                list(vec![value(Kind::Bool(true)), value(Kind::Bool(false))]),
            ),
        ]);
        assert_eq!(lane_final_info(Some(&masked_out), 1, 2), None);
    }

    #[test]
    fn phases_from_a_peer_that_stamps_nothing_read_back_as_zero() {
        use super::{EndpointPhases, env, model};

        let env_response = env::v1::JoinResponse {
            endpoint_total_ns: Some(1_000),
            ..Default::default()
        };
        assert_eq!(
            EndpointPhases::from_env_response(&env_response),
            EndpointPhases::default()
        );

        let model_response = model::v1::JoinResponse {
            endpoint_total_ns: Some(1_000),
            ..Default::default()
        };
        assert_eq!(
            EndpointPhases::from_model_response(&model_response),
            EndpointPhases::default()
        );

        let stamped = model::v1::JoinResponse {
            endpoint_total_ns: Some(1_000),
            decode_ns: Some(10),
            user_ns: Some(20),
            encode_ns: Some(30),
            queue_ns: Some(40),
            in_flight: Some(3),
            adapter_ns: Some(15),
            held_episodes: Some(5),
            held_state_bytes: Some(6_000),
            ..Default::default()
        };
        assert_eq!(
            EndpointPhases::from_model_response(&stamped),
            EndpointPhases {
                decode_ns: 10,
                user_ns: 20,
                encode_ns: 30,
                queue_ns: 40,
                in_flight: 3,
                adapter_ns: 15,
                held_episodes: Some(5),
                held_state_bytes: Some(6_000),
                lane_skew_ns: None,
            }
        );

        let skewed = env::v1::JoinResponse {
            endpoint_total_ns: Some(1_000),
            lane_skew_ns: Some(880),
            ..Default::default()
        };
        assert_eq!(
            EndpointPhases::from_env_response(&skewed).lane_skew_ns,
            Some(880)
        );
        let even = env::v1::JoinResponse {
            lane_skew_ns: Some(0),
            ..Default::default()
        };
        assert_eq!(
            EndpointPhases::from_env_response(&even).lane_skew_ns,
            Some(0)
        );

        assert_eq!(EndpointPhases::reported(0), None);
        assert_eq!(EndpointPhases::reported(7), Some(7));
    }

    #[test]
    fn nesting_folds_an_inner_split_into_the_outer_one() {
        use super::EndpointPhases;

        assert_eq!(
            EndpointPhases::nest(1, 10, 2, EndpointPhases::default()),
            EndpointPhases {
                decode_ns: 1,
                user_ns: 10,
                encode_ns: 2,
                ..EndpointPhases::default()
            }
        );

        let inner = EndpointPhases {
            decode_ns: 3,
            user_ns: 5,
            encode_ns: 2,
            ..EndpointPhases::default()
        };
        assert_eq!(
            EndpointPhases::nest(1, 10, 2, inner),
            EndpointPhases {
                decode_ns: 4,
                user_ns: 5,
                encode_ns: 4,
                ..EndpointPhases::default()
            }
        );

        assert_eq!(
            EndpointPhases::nest(1, 14, 2, inner),
            EndpointPhases {
                decode_ns: 4,
                user_ns: 9,
                encode_ns: 4,
                ..EndpointPhases::default()
            }
        );

        assert_eq!(
            EndpointPhases::nest(1, 4, 2, inner),
            EndpointPhases {
                decode_ns: 1,
                user_ns: 4,
                encode_ns: 2,
                ..EndpointPhases::default()
            }
        );
    }

    #[test]
    fn lane_skew_is_the_slowest_lane_over_the_median_one() {
        use super::lane_skew_ns;

        assert_eq!(lane_skew_ns(&mut [10, 10, 10, 10, 10, 10, 10, 900]), 890);
        assert_eq!(lane_skew_ns(&mut [900; 8]), 0);
        assert_eq!(lane_skew_ns(&mut [10, 100]), 90);
        assert_eq!(lane_skew_ns(&mut [1, 100, 100, 100]), 0);
        assert_eq!(lane_skew_ns(&mut [42]), 0);
        assert_eq!(lane_skew_ns(&mut []), 0);
    }

    #[test]
    fn nesting_keeps_the_inner_envs_lane_skew() {
        use super::EndpointPhases;

        let measured = EndpointPhases {
            user_ns: 5,
            lane_skew_ns: Some(77),
            ..EndpointPhases::default()
        };
        assert_eq!(
            EndpointPhases::nest(1, 10, 2, measured).lane_skew_ns,
            Some(77)
        );

        let skew_only = EndpointPhases {
            lane_skew_ns: Some(77),
            ..EndpointPhases::default()
        };
        assert_eq!(
            EndpointPhases::nest(1, 10, 2, skew_only).lane_skew_ns,
            Some(77)
        );
    }

    #[test]
    fn peer_info_default_then_override_merges_python_with_rust_fallback() {
        use super::{PeerInfoOverride, peer_info, set_peer_info_override};
        use std::collections::HashMap;

        let rust_info = peer_info("rlmesh-env");
        assert_eq!(rust_info.component, "rlmesh-env");
        assert_eq!(rust_info.language, "rust");
        assert!(rust_info.language_version.is_empty());
        assert!(rust_info.framework_versions.is_empty());
        let detected_os = rust_info.os.clone();
        let detected_arch = rust_info.arch.clone();
        let detected_pkg = rust_info.package_version.clone();

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
            extra: HashMap::from([("rlmesh.startup.listen_ms".to_string(), "4200".to_string())]),
        });

        let py_info = peer_info("rlmesh-env");
        assert_eq!(
            py_info
                .extra
                .get("rlmesh.startup.listen_ms")
                .map(String::as_str),
            Some("4200")
        );
        assert!(py_info.extra.contains_key("rlmesh.build.source"));
        assert_eq!(py_info.component, "rlmesh-env");
        assert_eq!(py_info.language, "python");
        assert_eq!(py_info.language_version, "3.11.4");
        assert_eq!(py_info.os_version, "ubuntu-22.04");
        assert_eq!(py_info.arch, "aarch64");
        assert_eq!(
            py_info.framework_versions.get("numpy").map(String::as_str),
            Some("1.26.4")
        );
        assert_eq!(py_info.os, detected_os);
        assert_eq!(py_info.package_version, detected_pkg);
        assert_eq!(
            py_info
                .extra
                .get("rlmesh.workflow.edition")
                .map(String::as_str),
            Some(CURRENT_WORKFLOW_EDITION)
        );
        let _ = detected_arch;
    }

    #[test]
    fn has_capability_reads_advertised_features() {
        use super::{capabilities, capability_map, has_capability};
        let map = capability_map(&[capabilities::MODEL_CONCURRENT_PREDICT_V1]);
        assert!(has_capability(
            &map,
            capabilities::MODEL_CONCURRENT_PREDICT_V1
        ));
        assert!(!has_capability(&map, "rlmesh.not.advertised.v1"));
        assert_eq!(map[capabilities::MODEL_CONCURRENT_PREDICT_V1], "true");
        for value in ["1", "yes", "TRUE", ""] {
            let odd =
                std::collections::HashMap::from([("rlmesh.odd.v1".to_string(), value.to_string())]);
            assert!(!has_capability(&odd, "rlmesh.odd.v1"), "{value:?}");
        }
    }

    #[test]
    fn env_subset_step_keeps_its_shipped_wire_spelling() {
        assert_eq!(super::capabilities::ENV_SUBSET_STEP, "subset_step");
    }
}
