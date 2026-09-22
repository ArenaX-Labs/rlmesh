//! The runtime session spec, its limits, and the report a finished run returns.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::Duration;

use rlmesh_proto::core::v1::{AutoresetMode, EnvContract};
use rlmesh_proto::spaces::v1::meta_value::Kind as MetaKind;
use rlmesh_proto::spaces::v1::{MetaList, MetaMap, MetaValue, SpaceSpec};
use rlmesh_proto::{Edition, EditionDefaults};
use rlmesh_spaces::{Advisory, DType};
use serde::{Deserialize, Serialize};

/// Empty fallback returned by the internal `*_validated` accessors only on the
/// unreachable path where the space is absent despite validation (see their
/// `debug_assert!`s). Lets those accessors stay panic-free and lint-clean.
static EMPTY_SPACE_SPEC: LazyLock<SpaceSpec> = LazyLock::new(SpaceSpec::default);

/// Key under which an env declares, in its contract metadata, which reserved
/// reset-option keys it wants delivered (a list of strings). The runtime sends a
/// reserved `ResetRequest.options` key only to an env that named it here, so an
/// env that forwards `options` blindly into a third-party `reset` never receives
/// one it cannot interpret. Mirrored in Python as `rlmesh.ENV_RESET_OPTIONS_KEY`.
pub const ENV_RESET_OPTIONS_KEY: &str = "rlmesh.env.v1.reset_options";

/// The reserved reset option carrying the ordinal of the episode a reset
/// starts. Declared by an env under [`ENV_RESET_OPTIONS_KEY`]; read in Python
/// as `rlmesh.trial_index(options)`.
///
/// The env-side spelling of the key. The driver never reads it: a session takes
/// its trial-index key from [`EditionDefaults::trial_index_option_key`], so the
/// edition governs what goes on the wire.
pub const TRIAL_INDEX_OPTION: &str = "trial_index";

/// Whether `contract` named `key` in its metadata under
/// [`ENV_RESET_OPTIONS_KEY`], as either a list of strings or a bare string.
///
/// The declaration gate for every reserved `ResetRequest.options` key: an env
/// that forwards `options` blindly into a third-party `reset` must never
/// receive a reserved key it cannot interpret.
pub fn declares_reset_option(contract: &EnvContract, key: &str) -> bool {
    let Some(declared) = contract
        .spec
        .as_ref()
        .and_then(|spec| spec.metadata.as_ref())
        .and_then(|metadata| metadata.entries.get(ENV_RESET_OPTIONS_KEY))
        .and_then(|declared| declared.kind.as_ref())
    else {
        return false;
    };
    match declared {
        MetaKind::List(list) => list.items.iter().any(
            |item| matches!(item.kind.as_ref(), Some(MetaKind::Text(declared)) if declared == key),
        ),
        MetaKind::Text(declared) => declared == key,
        _ => false,
    }
}

/// The `ResetRequest.options` map delivering `trials` under `option_key` — the
/// trial-index key the session's edition governs
/// ([`EditionDefaults::trial_index_option_key`], mirrored for env-side callers by
/// [`TRIAL_INDEX_OPTION`]) — or `None` when there are no trials or the env never
/// declared the key.
///
/// A single lane sends the bare integer; a multi-lane reset sends the list, in
/// the same lane order as `seeds` and `episode_ids`.
pub fn reset_options_for(
    contract: &EnvContract,
    option_key: &str,
    trials: &[u64],
) -> Option<MetaMap> {
    if trials.is_empty() || !declares_reset_option(contract, option_key) {
        return None;
    }
    let value = if trials.len() == 1 {
        MetaValue {
            kind: Some(MetaKind::Integer(trials[0] as i64)),
        }
    } else {
        MetaValue {
            kind: Some(MetaKind::List(MetaList {
                items: trials
                    .iter()
                    .map(|trial| MetaValue {
                        kind: Some(MetaKind::Integer(*trial as i64)),
                    })
                    .collect(),
            })),
        }
    };
    Some(MetaMap {
        entries: [(option_key.to_string(), value)].into_iter().collect(),
    })
}

/// What one served peer can decode, learned at its handshake. The driver checks
/// every payload it relays against the target leg's ceiling through the
/// session's [`RelayPolicy`](crate::RelayPolicy), not against the session
/// edition alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerCeiling {
    /// The highest edition this peer and this build both implement, which may
    /// exceed the session edition
    /// ([`workflow_edition`](RuntimeSessionSpec::workflow_edition)).
    pub edition: Edition,
    pub dtypes: HashSet<DType>,
    /// What the peer advertised; query with [`rlmesh_proto::has_capability`].
    pub capabilities: HashMap<String, String>,
    /// The largest message the peer accepts, in bytes.
    pub max_message_size: usize,
}

impl PeerCeiling {
    /// The highest edition this build retains among a peer's advertised
    /// `supported` names, or `None` when they share none.
    pub fn highest_shared_edition(supported: &[String]) -> Option<Edition> {
        supported
            .iter()
            .filter(|name| rlmesh_proto::parse_retained_edition(name).is_ok())
            .max_by(|a, b| {
                rlmesh_proto::edition_sort_key(a).cmp(&rlmesh_proto::edition_sort_key(b))
            })
            .and_then(|name| rlmesh_proto::parse_retained_edition(name).ok())
    }

    /// A wire-v1 peer's ceiling. Wire-v1 advertises no dtype set, so a peer
    /// decodes every dtype the wire defines.
    pub fn wire_v1(
        edition: Edition,
        capabilities: HashMap<String, String>,
        max_message_size: usize,
    ) -> Self {
        Self {
            edition,
            dtypes: DType::ALL.into_iter().collect(),
            capabilities,
            max_message_size,
        }
    }
}

/// Everything one route needs to run: its identity, the negotiated env
/// contract, and the per-op limits. [`validate`](Self::validate) gates a spec
/// before the driver runs it.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeSessionSpec {
    /// Correlation label only; OSS does not key on it (the managed layer owns
    /// session lifecycle). Kept as plumbing for telemetry/logs — its removal is
    /// the deferred closed split, out of scope here.
    pub session_id: String,
    /// The connected env container, UUIDv7 (minted by the runtime on attach).
    /// The single routing key: replaces the old `route_id` + positional lane.
    /// (Repurposed from the former descriptive-name field; the human env name
    /// now lives only in the language SDK's own contract type.)
    pub env_id: String,
    pub env_component_id: String,
    pub model_component_id: String,
    /// Workflow edition negotiated at the env handshake, already resolved to the
    /// typed arm whose semantics this session runs under. Wire names are parsed
    /// once, where they arrive, by
    /// [`rlmesh_proto::parse_retained_edition`] — that is where the runtime
    /// refuses an edition it was not built to drive, so a name outside the
    /// retained list never reaches this field.
    pub workflow_edition: Edition,
    pub env_contract: EnvContract,
    pub num_envs: usize,
    pub base_seed: Option<i64>,
    /// Explicit per-episode reset seeds, consumed in episode-start order (a
    /// vector reset claims one per lane). Overrides `base_seed` derivation when
    /// non-empty; episodes beyond the list reset unseeded. Requires
    /// driver-owned resets (autoreset `DISABLED`): under `NEXT_STEP` the env
    /// seeds its own rolls, so the list would silently not apply.
    pub episode_seeds: Vec<i64>,
    pub max_episodes: Option<u64>,
    /// First trial ordinal this route's episodes walk; `None` is the default
    /// base, 0 (so is an explicit `Some(0)`). Under driver-owned resets
    /// (autoreset `DISABLED`) the driver always mints one ordinal per episode
    /// start (`base`, `base + 1`, ...), reports it on the episode events and
    /// summaries, and delivers it as `ResetRequest.options["trial_index"]` to an
    /// env that declared the key (see [`ENV_RESET_OPTIONS_KEY`]) -- an env that
    /// did not never sees it. A sharded run gives each shard its own base so the
    /// shards together walk a benchmark's trials once each, instead of every
    /// shard re-deriving an index from a hashed seed. Under `NEXT_STEP` autoreset
    /// the env restarts its own lanes, so no ordinal is minted at all: the
    /// default base is inert there, and a non-zero base is rejected by
    /// [`validate`](Self::validate) since it could never be walked.
    pub trial_index_base: Option<u64>,
    /// Truncate any episode after this many steps (runtime-enforced; the lane
    /// is reset and the episode reported `truncated`). Requires driver-owned
    /// resets (autoreset `DISABLED`).
    pub max_episode_steps: Option<i64>,
    /// Truncate any episode after this wall-clock duration (seconds), same
    /// semantics and autoreset requirement as `max_episode_steps`.
    pub max_episode_seconds: Option<f64>,
    pub close_env_on_end: bool,
    pub limits: RuntimeLimits,
    /// The env advertised the `subset_step` handshake capability: every lane
    /// can be reset and stepped on its own. The driver then runs one episode
    /// loop per lane (a lane is its own group) instead of stepping the vector
    /// in lockstep, and episode seeds/indices come from a route-global slot
    /// counter so the scored set is fixed by the budget alone.
    pub subset_step: bool,
    /// The served env's ceiling; `None` when the env runs in-process.
    pub env_ceiling: Option<PeerCeiling>,
    /// The served model's ceiling; `None` when the model runs in-process.
    pub model_ceiling: Option<PeerCeiling>,
}

impl RuntimeSessionSpec {
    /// The first trial ordinal this route walks: `trial_index_base`, or 0 when
    /// the session left it unset.
    pub fn trial_index_base(&self) -> u64 {
        self.trial_index_base.unwrap_or(0)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.session_id.trim().is_empty() {
            return Err("runtime session_id must not be empty".to_string());
        }
        if self.env_id.trim().is_empty() {
            return Err("runtime env_id must not be empty".to_string());
        }
        if self.env_component_id.trim().is_empty() {
            return Err("runtime env_component_id must not be empty".to_string());
        }
        if self.model_component_id.trim().is_empty() {
            return Err("runtime model_component_id must not be empty".to_string());
        }
        if self.num_envs == 0 {
            return Err("runtime num_envs must be greater than zero".to_string());
        }
        if self.observation_space().is_none() {
            return Err("runtime env_contract is missing observation_space".to_string());
        }
        if self.action_space().is_none() {
            return Err("runtime env_contract is missing action_space".to_string());
        }
        if self.max_episodes == Some(0) {
            return Err("runtime max_episodes must be greater than zero when set".to_string());
        }
        if self.max_episode_steps.is_some_and(|cap| cap <= 0) {
            return Err("runtime max_episode_steps must be greater than zero when set".to_string());
        }
        if self.max_episode_seconds.is_some_and(|cap| cap <= 0.0) {
            return Err(
                "runtime max_episode_seconds must be greater than zero when set".to_string(),
            );
        }
        let driver_owns_resets = AutoresetMode::try_from(self.env_contract.autoreset_mode)
            .is_ok_and(|mode| {
                self.edition_defaults()
                    .driver_owned_reset_modes
                    .contains(&mode)
            });
        if !driver_owns_resets {
            if !self.episode_seeds.is_empty() {
                return Err(
                    "episode_seeds requires an env with autoreset disabled: under NEXT_STEP \
                     autoreset the env seeds its own episode rolls, so explicit per-episode \
                     seeds cannot apply"
                        .to_string(),
                );
            }
            if self.trial_index_base() != 0 {
                return Err(
                    "trial_index_base requires an env with autoreset disabled: under \
                     NEXT_STEP autoreset the env restarts its own lanes, so the runtime \
                     has no reset on which to deliver a trial ordinal"
                        .to_string(),
                );
            }
            if self.max_episode_steps.is_some() || self.max_episode_seconds.is_some() {
                return Err(
                    "max_episode_steps / max_episode_seconds require an env with autoreset \
                     disabled: under NEXT_STEP autoreset the env owns lane resets, so the \
                     runtime cannot truncate an episode"
                        .to_string(),
                );
            }
        }
        // The runtime drives one of the editions it was built for. Wire names are
        // already refused at the parse boundary, so this re-checks the typed value
        // for the one route that skips it: every field here is public, so a caller
        // can name an arm this build implements but no longer retains. Same
        // authority, same wording — membership, not equality with CURRENT, since a
        // retained older edition (a graceful downgrade) is a valid session floor.
        rlmesh_proto::parse_retained_edition(self.workflow_edition.base())?;
        // An autoreset mode this build does not understand (e.g. a newer peer's
        // mode) must fail loudly at session setup, never silently fold to
        // DISABLED and change lifecycle semantics.
        if AutoresetMode::try_from(self.env_contract.autoreset_mode).is_err() {
            return Err(format!(
                "unknown autoreset mode {} on the wire; this build supports \
                 UNSPECIFIED, NEXT_STEP, SAME_STEP, DISABLED only",
                self.env_contract.autoreset_mode
            ));
        }
        // SAME_STEP is reserved on the wire but not yet driven by the runtime:
        // the driver currently aliases NEXT_STEP|SAME_STEP to a purely
        // observational path, while the env server never rolls SAME_STEP episode
        // ids -> done lanes would stall. Reject it here so it cannot reach the
        // runtime under a false assumption of support.
        if self.env_contract.autoreset_mode == AutoresetMode::SameStep as i32 {
            return Err(
                "SAME_STEP autoreset is reserved but not yet supported by the runtime; \
                 construct the env with NEXT_STEP or DISABLED autoreset"
                    .to_string(),
            );
        }
        // A lockstep vectorized session (one group of N lanes) requires
        // NEXT_STEP autoreset: the env resets each done lane itself. Under
        // DISABLED the driver would have to reset just the done lanes, which a
        // stock gymnasium vector env cannot do (a full reset clobbers the
        // still-running lanes). A lane endpoint (`subset_step`) is driven one
        // group per lane instead, where DISABLED is the norm. Reject the
        // combination up front instead of failing mid-run the first time lanes
        // terminate at different steps. (SAME_STEP is already rejected above.)
        if self.num_envs > 1
            && !self.subset_step
            && self.env_contract.autoreset_mode != AutoresetMode::NextStep as i32
        {
            return Err(
                "vectorized runtime sessions (num_envs > 1) require NEXT_STEP autoreset unless \
                 the env steps lanes individually (the `subset_step` capability); DISABLED \
                 autoreset needs per-lane reset, which a stock gymnasium vector env cannot do. \
                 Use NEXT_STEP autoreset, serve lanes, or run with num_envs == 1."
                    .to_string(),
            );
        }
        Ok(())
    }

    /// The edition-governed defaults this session runs under. Every value the
    /// driver would otherwise hardcode comes from here, so a future edition
    /// changes a table row instead of a code path.
    pub fn edition_defaults(&self) -> &'static EditionDefaults {
        rlmesh_proto::defaults(self.workflow_edition)
    }

    pub fn env_context(&self) -> crate::hooks::RuntimeEnvContext {
        crate::hooks::RuntimeEnvContext {
            env_id: self.env_id.clone(),
            env_component_id: self.env_component_id.clone(),
            model_component_id: self.model_component_id.clone(),
            lane: None,
        }
    }

    /// Returns the observation space, or `None` if the spec has not been
    /// populated/validated (`env_contract.observation_space` is unset).
    ///
    /// All `RuntimeSessionSpec` fields are public, so an unvalidated spec is
    /// trivial to construct; this accessor never panics. The driver validates
    /// the spec before running and uses the infallible internal accessor.
    pub fn observation_space(&self) -> Option<&SpaceSpec> {
        self.env_contract
            .spec
            .as_ref()
            .and_then(|spec| spec.observation_space.as_ref())
    }

    /// Returns the action space, or `None` if the spec has not been
    /// populated/validated (`env_contract.action_space` is unset).
    ///
    /// See [`RuntimeSessionSpec::observation_space`] for why this is fallible.
    pub fn action_space(&self) -> Option<&SpaceSpec> {
        self.env_contract
            .spec
            .as_ref()
            .and_then(|spec| spec.action_space.as_ref())
    }

    /// Observation space for internal use after [`validate`](Self::validate)
    /// has confirmed it is present.
    pub(crate) fn observation_space_validated(&self) -> &SpaceSpec {
        debug_assert!(
            self.observation_space().is_some(),
            "observation_space accessed before validate()"
        );
        // LazyLock<SpaceSpec> derefs to &SpaceSpec on the unreachable None path.
        self.observation_space()
            .unwrap_or_else(|| &EMPTY_SPACE_SPEC)
    }

    /// Action space for internal use after [`validate`](Self::validate) has
    /// confirmed it is present.
    pub(crate) fn action_space_validated(&self) -> &SpaceSpec {
        debug_assert!(
            self.action_space().is_some(),
            "action_space accessed before validate()"
        );
        // LazyLock<SpaceSpec> derefs to &SpaceSpec on the unreachable None path.
        self.action_space().unwrap_or_else(|| &EMPTY_SPACE_SPEC)
    }
}

/// One completed episode's summary, recorded in completion order across all
/// lanes. The pull counterpart of the `episode_completed` hook event, so a
/// caller without hooks (e.g. the in-process `run_local` loop) still gets
/// per-episode results on the report.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeSummary {
    /// 1-based slot ordinal within the session, in completion order.
    pub episode_index: i64,
    /// The vector lane the episode ran on (0 for a single env).
    pub env_index: i32,
    /// The explicit seed this episode was reset with (`episode_seeds` /
    /// `base_seed` derivation), `None` for an unseeded or autoreset-rolled one.
    pub seed: Option<i64>,
    /// The trial ordinal this episode walked (`trial_index_base` + its
    /// episode-start position). Minted for every driver-owned reset whether or
    /// not the env declared the reset option, so a coverage audit can read the
    /// sweep off the report either way; `None` only under `NEXT_STEP` autoreset,
    /// where the env restarts its own lanes and no ordinal is minted.
    pub trial_index: Option<u64>,
    pub step_count: i64,
    pub cumulative_reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub duration_ms: i64,
    /// Env-reported task outcome from the final step's info (Gymnasium's
    /// `is_success` / `success`, or `task_success`); `None` when the env emits
    /// no such signal.
    pub success: Option<bool>,
}

/// What a finished or aborted session returns: totals plus the durable
/// telemetry aggregate.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeReport {
    pub session_id: String,
    pub env_id: String,
    pub total_steps: i64,
    pub total_episodes: i64,
    /// Every completed episode, in completion order.
    pub episodes: Vec<EpisodeSummary>,
    /// Session-total telemetry aggregate (per-op latency/percentiles/bytes) —
    /// the durable pull counterpart to the live `RuntimeHooks::on_telemetry` push.
    pub telemetry: crate::telemetry::Snapshot,
    /// Each distinct advisory the relay policy raised, in first-raised order.
    pub advisories: Vec<Advisory>,
}

/// Per-op timeouts and the telemetry window for one session. Serialized with
/// explicit millisecond field names (`*Ms`); legacy unsuffixed fields are rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RuntimeLimits {
    #[serde(
        default = "default_connect_timeout",
        rename = "envConnectTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub env_connect_timeout: Duration,
    #[serde(
        default = "default_model_connect_timeout",
        rename = "modelConnectTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub model_connect_timeout: Duration,
    #[serde(
        default = "default_configure_route_timeout",
        rename = "configureRouteTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub configure_route_timeout: Duration,
    #[serde(
        default = "default_env_reset_timeout",
        rename = "envResetTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub env_reset_timeout: Duration,
    #[serde(
        default = "default_model_predict_timeout",
        rename = "modelPredictTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub model_predict_timeout: Duration,
    #[serde(
        default = "default_env_step_timeout",
        rename = "envStepTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub env_step_timeout: Duration,
    #[serde(
        default = "default_service_close_timeout",
        rename = "serviceCloseTimeoutMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub service_close_timeout: Duration,
    /// How often the background ticker pushes Window + Session telemetry
    /// snapshots to `on_telemetry`, on a wall clock. `0` disables live streaming
    /// (the final session snapshot is still delivered at session end); any
    /// non-zero value below 1ms is floored to 1ms.
    #[serde(
        default = "default_telemetry_window",
        rename = "telemetryWindowMs",
        serialize_with = "duration_millis::serialize",
        deserialize_with = "duration_millis::deserialize"
    )]
    pub telemetry_window: Duration,
}

impl Default for RuntimeLimits {
    fn default() -> Self {
        Self {
            env_connect_timeout: default_connect_timeout(),
            model_connect_timeout: default_model_connect_timeout(),
            configure_route_timeout: default_configure_route_timeout(),
            env_reset_timeout: default_env_reset_timeout(),
            model_predict_timeout: default_model_predict_timeout(),
            env_step_timeout: default_env_step_timeout(),
            service_close_timeout: default_service_close_timeout(),
            telemetry_window: default_telemetry_window(),
        }
    }
}

impl RuntimeLimits {
    /// Clamped-non-negative i64 milliseconds; the proto field is uint64, so
    /// callers `.max(0) as u64` without losing information.
    pub fn env_step_timeout_ms(&self) -> i64 {
        duration_ms_i64(self.env_step_timeout)
    }

    /// Clamped-non-negative i64 milliseconds; the proto field is uint64, so
    /// callers `.max(0) as u64` without losing information.
    pub fn env_reset_timeout_ms(&self) -> i64 {
        duration_ms_i64(self.env_reset_timeout)
    }
}

fn default_connect_timeout() -> Duration {
    Duration::from_secs(60)
}

fn default_model_connect_timeout() -> Duration {
    Duration::from_secs(600)
}

fn default_configure_route_timeout() -> Duration {
    Duration::from_secs(600)
}

fn default_env_reset_timeout() -> Duration {
    Duration::from_secs(300)
}

fn default_model_predict_timeout() -> Duration {
    Duration::from_secs(300)
}

fn default_env_step_timeout() -> Duration {
    Duration::from_secs(300)
}

fn default_service_close_timeout() -> Duration {
    Duration::from_secs(5)
}

fn default_telemetry_window() -> Duration {
    Duration::from_secs(1)
}

fn duration_ms_i64(duration: Duration) -> i64 {
    duration.as_millis().try_into().unwrap_or(i64::MAX)
}

mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub(super) fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let millis = duration.as_millis().try_into().unwrap_or(u64::MAX);
        serializer.serialize_u64(millis)
    }

    pub(super) fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Duration::from_millis(u64::deserialize(deserializer)?))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rlmesh_proto::core::v1::{AutoresetMode, EnvContract, EnvSpec};
    use rlmesh_proto::spaces::v1::SpaceSpec;
    use serde_json::json;

    use super::{
        ENV_RESET_OPTIONS_KEY, MetaKind, MetaList, MetaMap, MetaValue, RuntimeLimits,
        RuntimeSessionSpec, TRIAL_INDEX_OPTION, declares_reset_option, reset_options_for,
    };

    fn valid_spec() -> RuntimeSessionSpec {
        RuntimeSessionSpec {
            session_id: "session".to_string(),
            env_id: "env-id".to_string(),
            env_component_id: "env".to_string(),
            model_component_id: "model".to_string(),
            workflow_edition: rlmesh_proto::parse_retained_edition(
                rlmesh_proto::CURRENT_WORKFLOW_EDITION,
            )
            .expect("this build drives its own edition"),
            env_contract: EnvContract {
                spec: Some(EnvSpec {
                    observation_space: Some(SpaceSpec::default()),
                    action_space: Some(SpaceSpec::default()),
                    ..Default::default()
                }),
                num_envs: 1,
                ..Default::default()
            },
            num_envs: 1,
            episode_seeds: Vec::new(),
            base_seed: None,
            max_episodes: Some(1),
            trial_index_base: None,
            max_episode_steps: None,
            max_episode_seconds: None,
            close_env_on_end: true,
            subset_step: false,
            limits: RuntimeLimits::default(),
            env_ceiling: None,
            model_ceiling: None,
        }
    }

    #[test]
    fn validate_rejects_a_trial_base_the_env_owns_the_resets_for() {
        let mut spec = valid_spec();
        spec.trial_index_base = Some(5);
        // Driver-owned resets (the default DISABLED/UNSPECIFIED) carry the ordinal.
        assert!(spec.validate().is_ok());

        spec.env_contract.autoreset_mode = AutoresetMode::NextStep as i32;
        let error = spec.validate().unwrap_err();
        assert!(
            error.contains("trial_index_base requires an env with autoreset disabled"),
            "expected the driver-owned-reset rule, got: {error}"
        );
    }

    #[test]
    fn validate_accepts_the_default_trial_base_under_next_step() {
        // The ordinal is on by default, so the default base must not fail a run
        // the env owns the resets for -- whether it arrives unset or as an
        // explicit 0 (the Python surface always passes an integer).
        let mut spec = valid_spec();
        spec.env_contract.autoreset_mode = AutoresetMode::NextStep as i32;
        for base in [None, Some(0)] {
            spec.trial_index_base = base;
            assert_eq!(spec.trial_index_base(), 0);
            assert!(spec.validate().is_ok(), "base {base:?} must validate");
        }
    }

    #[test]
    fn reset_options_key_is_the_published_string() {
        // Pinned against the Python mirror (`rlmesh.ENV_RESET_OPTIONS_KEY`, itself
        // re-exported from this constant) by
        // `python/rlmesh/tests/unit/test_reset_options.py`.
        assert_eq!(super::ENV_RESET_OPTIONS_KEY, "rlmesh.env.v1.reset_options");
    }

    #[test]
    fn a_ceiling_edition_is_the_highest_one_both_sides_implement() {
        let names = |names: &[&str]| {
            names
                .iter()
                .map(|name| name.to_string())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            super::PeerCeiling::highest_shared_edition(&names(&[
                "2099.01",
                rlmesh_proto::CURRENT_WORKFLOW_EDITION,
            ])),
            Some(rlmesh_proto::Edition::current())
        );
        assert_eq!(
            super::PeerCeiling::highest_shared_edition(&names(&["2099.01"])),
            None
        );
    }

    /// The spec carries a typed edition, so a made-up name is refused one step
    /// earlier — at the shared parse boundary the env client and every other
    /// string-carrying caller go through — naming the arrived string and the
    /// retained list. That refusal is what an operator actually sees.
    #[test]
    fn an_edition_the_runtime_cannot_drive_is_refused_by_name() {
        let error = rlmesh_proto::parse_retained_edition("2099.01")
            .expect_err("an unimplemented edition is refused");
        assert!(
            error.contains("2099.01")
                && error.contains("cannot drive")
                && error.contains(rlmesh_proto::CURRENT_WORKFLOW_EDITION),
            "expected an edition-refusal error naming the retained list, got: {error}"
        );

        // The edition the build implements is accepted, by base name and by this
        // build's cohort spelling of it, and validates.
        let mut spec = valid_spec();
        for name in [
            rlmesh_proto::CURRENT_WORKFLOW_EDITION,
            rlmesh_proto::WORKFLOW_EDITION_BASE,
        ] {
            spec.workflow_edition = rlmesh_proto::parse_retained_edition(name)
                .unwrap_or_else(|error| panic!("{name} must parse: {error}"));
            assert!(spec.validate().is_ok());
        }
    }

    #[test]
    fn space_accessors_return_none_on_unvalidated_spec() {
        let mut spec = valid_spec();
        // An unvalidated spec is trivially constructible since all fields are
        // public; the accessors must not panic.
        spec.env_contract = EnvContract::default();

        assert!(spec.observation_space().is_none());
        assert!(spec.action_space().is_none());
    }

    #[test]
    fn space_accessors_return_some_on_populated_spec() {
        let spec = valid_spec();
        assert!(spec.observation_space().is_some());
        assert!(spec.action_space().is_some());
    }

    #[test]
    fn validate_accepts_vectorized_next_step_runtime_sessions() {
        // num_envs > 1 is supported with NEXT_STEP autoreset: the env resets each
        // done lane itself, so the driver never needs per-lane reset.
        let mut spec = valid_spec();
        spec.num_envs = 4;
        spec.env_contract.num_envs = 4;
        spec.env_contract.autoreset_mode = AutoresetMode::NextStep as i32;

        assert!(spec.validate().is_ok());
    }

    #[test]
    fn validate_rejects_disabled_vectorized_sessions() {
        // DISABLED (and the UNSPECIFIED default) with num_envs > 1 needs per-lane
        // reset, which stock gymnasium vector envs cannot do. Reject up front
        // rather than failing mid-run on the first staggered termination.
        for mode in [AutoresetMode::Disabled, AutoresetMode::Unspecified] {
            let mut spec = valid_spec();
            spec.num_envs = 4;
            spec.env_contract.num_envs = 4;
            spec.env_contract.autoreset_mode = mode as i32;

            let error = spec.validate().unwrap_err();
            assert!(
                error.contains("NEXT_STEP"),
                "expected a NEXT_STEP-guidance rejection for {mode:?}, got: {error}"
            );
        }
    }

    #[test]
    fn validate_rejects_same_step_autoreset() {
        // SAME_STEP is reserved but unsupported; validation must reject it so it
        // cannot reach the runtime and stall lanes.
        let mut spec = valid_spec();
        spec.env_contract.autoreset_mode = AutoresetMode::SameStep as i32;

        let error = spec.validate().unwrap_err();
        assert!(
            error.contains("SAME_STEP"),
            "expected SAME_STEP rejection, got: {error}"
        );
    }

    #[test]
    fn runtime_limits_json_uses_explicit_millisecond_fields() {
        let value = serde_json::to_value(RuntimeLimits::default()).unwrap();

        assert_eq!(value["envConnectTimeoutMs"], json!(60_000));
        assert_eq!(value["modelConnectTimeoutMs"], json!(600_000));
        assert_eq!(value["configureRouteTimeoutMs"], json!(600_000));
        assert_eq!(value["envResetTimeoutMs"], json!(300_000));
        assert_eq!(value["modelPredictTimeoutMs"], json!(300_000));
        assert_eq!(value["envStepTimeoutMs"], json!(300_000));
        assert_eq!(value["serviceCloseTimeoutMs"], json!(5_000));
        assert_eq!(value["telemetryWindowMs"], json!(1_000));
        assert!(value.get("envConnectTimeout").is_none());

        let parsed: RuntimeLimits = serde_json::from_value(json!({
            "envConnectTimeoutMs": 1,
            "modelConnectTimeoutMs": 2,
            "configureRouteTimeoutMs": 3,
            "envResetTimeoutMs": 4,
            "modelPredictTimeoutMs": 5,
            "envStepTimeoutMs": 6,
            "serviceCloseTimeoutMs": 7,
            "telemetryWindowMs": 8
        }))
        .unwrap();

        assert_eq!(parsed.env_connect_timeout, Duration::from_millis(1));
        assert_eq!(parsed.model_connect_timeout, Duration::from_millis(2));
        assert_eq!(parsed.configure_route_timeout, Duration::from_millis(3));
        assert_eq!(parsed.env_reset_timeout, Duration::from_millis(4));
        assert_eq!(parsed.model_predict_timeout, Duration::from_millis(5));
        assert_eq!(parsed.env_step_timeout, Duration::from_millis(6));
        assert_eq!(parsed.service_close_timeout, Duration::from_millis(7));
        assert_eq!(parsed.telemetry_window, Duration::from_millis(8));
    }

    #[test]
    fn runtime_limits_reject_legacy_unsuffixed_fields() {
        let error = serde_json::from_value::<RuntimeLimits>(json!({
            "envConnectTimeout": 1
        }))
        .unwrap_err();

        assert!(error.to_string().contains("envConnectTimeout"));
    }

    /// An env contract whose metadata declares `reset_options = declared`.
    fn contract_declaring(declared: MetaValue) -> EnvContract {
        EnvContract {
            spec: Some(EnvSpec {
                metadata: Some(MetaMap {
                    entries: [(ENV_RESET_OPTIONS_KEY.to_string(), declared)].into(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn text(value: &str) -> MetaValue {
        MetaValue {
            kind: Some(MetaKind::Text(value.to_string())),
        }
    }

    fn list(items: Vec<MetaValue>) -> MetaValue {
        MetaValue {
            kind: Some(MetaKind::List(MetaList { items })),
        }
    }

    fn trial_option(options: &MetaMap) -> Option<&MetaKind> {
        options.entries.get(TRIAL_INDEX_OPTION)?.kind.as_ref()
    }

    #[test]
    fn declares_reset_option_reads_a_list_or_a_bare_string() {
        assert!(declares_reset_option(
            &contract_declaring(list(vec![text("other"), text(TRIAL_INDEX_OPTION)])),
            TRIAL_INDEX_OPTION,
        ));
        assert!(declares_reset_option(
            &contract_declaring(text(TRIAL_INDEX_OPTION)),
            TRIAL_INDEX_OPTION,
        ));
        assert!(!declares_reset_option(
            &contract_declaring(list(vec![text("other")])),
            TRIAL_INDEX_OPTION,
        ));
        assert!(!declares_reset_option(
            &contract_declaring(MetaValue {
                kind: Some(MetaKind::Integer(1)),
            }),
            TRIAL_INDEX_OPTION,
        ));
        assert!(!declares_reset_option(
            &EnvContract::default(),
            TRIAL_INDEX_OPTION,
        ));
    }

    #[test]
    fn reset_options_for_sends_an_integer_per_lane_and_a_list_for_many() {
        let contract = contract_declaring(list(vec![text(TRIAL_INDEX_OPTION)]));

        let single =
            reset_options_for(&contract, TRIAL_INDEX_OPTION, &[7]).expect("single-lane options");
        assert_eq!(trial_option(&single), Some(&MetaKind::Integer(7)));

        let many = reset_options_for(&contract, TRIAL_INDEX_OPTION, &[7, 8, 9])
            .expect("multi-lane options");
        assert_eq!(
            trial_option(&many),
            Some(&MetaKind::List(MetaList {
                items: [7, 8, 9]
                    .into_iter()
                    .map(|trial| MetaValue {
                        kind: Some(MetaKind::Integer(trial)),
                    })
                    .collect(),
            })),
        );
    }

    #[test]
    fn reset_options_for_withholds_without_trials_or_a_declaration() {
        let contract = contract_declaring(list(vec![text(TRIAL_INDEX_OPTION)]));

        assert_eq!(reset_options_for(&contract, TRIAL_INDEX_OPTION, &[]), None);
        assert_eq!(
            reset_options_for(
                &contract_declaring(list(vec![text("other")])),
                TRIAL_INDEX_OPTION,
                &[7]
            ),
            None,
        );
        assert_eq!(
            reset_options_for(&EnvContract::default(), TRIAL_INDEX_OPTION, &[7]),
            None
        );
    }
}
