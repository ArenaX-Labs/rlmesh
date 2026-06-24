use std::sync::LazyLock;
use std::time::Duration;

use rlmesh_proto::core::v1::{AutoresetMode, EnvContract};
use rlmesh_proto::spaces::v1::SpaceSpec;
use serde::{Deserialize, Serialize};

/// Empty fallback returned by the internal `*_validated` accessors only on the
/// unreachable path where the space is absent despite validation (see their
/// `debug_assert!`s). Lets those accessors stay panic-free and lint-clean.
static EMPTY_SPACE_SPEC: LazyLock<SpaceSpec> = LazyLock::new(SpaceSpec::default);

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeSessionSpec {
    pub session_id: String,
    pub route_id: String,
    pub env_component_id: String,
    pub model_component_id: String,
    pub env_id: String,
    /// Workflow edition negotiated at the env handshake. The runtime refuses an
    /// edition it was not built to drive (see [`RuntimeSessionSpec::validate`]).
    pub workflow_edition: String,
    pub env_contract: EnvContract,
    pub num_envs: usize,
    pub base_seed: Option<i64>,
    pub max_episodes: Option<u64>,
    pub close_env_on_end: bool,
    pub limits: RuntimeLimits,
}

impl RuntimeSessionSpec {
    pub fn validate(&self) -> Result<(), String> {
        if self.session_id.trim().is_empty() {
            return Err("runtime session_id must not be empty".to_string());
        }
        if self.route_id.trim().is_empty() {
            return Err("runtime route_id must not be empty".to_string());
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
        // The runtime drives exactly the workflow edition it was built for; a
        // session negotiated under any other edition is refused rather than run
        // under 2026.06 semantics it never agreed to.
        if self.workflow_edition != rlmesh_proto::CURRENT_WORKFLOW_EDITION {
            return Err(format!(
                "runtime cannot drive workflow edition {:?}; this build implements {:?}",
                self.workflow_edition,
                rlmesh_proto::CURRENT_WORKFLOW_EDITION
            ));
        }
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
        // Vectorized sessions require NEXT_STEP autoreset: the env resets each
        // done lane itself, so the driver never needs per-lane reset. DISABLED
        // (and the UNSPECIFIED default) would require resetting just the done
        // lanes, which stock gymnasium vector envs cannot do. There is no
        // partial-reset API, and a full reset clobbers the still-running lanes.
        // Reject the combination up front instead of failing mid-run the first
        // time lanes terminate at different steps. A future in-house vector
        // engine with per-lane reset will lift this gate. (SAME_STEP is already
        // rejected above, so the only mode that passes here is NEXT_STEP.)
        if self.num_envs > 1 && self.env_contract.autoreset_mode != AutoresetMode::NextStep as i32 {
            return Err(
                "vectorized runtime sessions (num_envs > 1) require NEXT_STEP autoreset; \
                 DISABLED autoreset needs per-lane reset, which is unavailable for stock \
                 gymnasium vector envs. Use NEXT_STEP autoreset, or run with num_envs == 1."
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn route_context(&self) -> crate::hooks::RuntimeRouteContext {
        crate::hooks::RuntimeRouteContext {
            route_id: self.route_id.clone(),
            env_component_id: self.env_component_id.clone(),
            model_component_id: self.model_component_id.clone(),
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

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeReport {
    pub session_id: String,
    pub route_id: String,
    pub total_steps: i64,
    pub total_episodes: i64,
}

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
        }
    }
}

impl RuntimeLimits {
    pub fn env_step_timeout_ms(&self) -> i64 {
        duration_ms_i64(self.env_step_timeout)
    }

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

    use super::{RuntimeLimits, RuntimeSessionSpec};

    fn valid_spec() -> RuntimeSessionSpec {
        RuntimeSessionSpec {
            session_id: "session".to_string(),
            route_id: "route".to_string(),
            env_component_id: "env".to_string(),
            model_component_id: "model".to_string(),
            env_id: "env-id".to_string(),
            workflow_edition: rlmesh_proto::CURRENT_WORKFLOW_EDITION.to_string(),
            env_contract: EnvContract {
                spec: Some(EnvSpec {
                    observation_space: Some(SpaceSpec::default()),
                    action_space: Some(SpaceSpec::default()),
                    ..Default::default()
                }),
                num_envs: Some(1),
                ..Default::default()
            },
            num_envs: 1,
            base_seed: None,
            max_episodes: Some(1),
            close_env_on_end: true,
            limits: RuntimeLimits::default(),
        }
    }

    #[test]
    fn validate_rejects_an_edition_the_runtime_cannot_drive() {
        let mut spec = valid_spec();
        spec.workflow_edition = "2099.01".to_string();
        let error = spec.validate().unwrap_err();
        assert!(
            error.contains("2099.01") && error.contains("cannot drive"),
            "expected an edition-refusal error, got: {error}"
        );

        // The edition the build implements is accepted.
        spec.workflow_edition = rlmesh_proto::CURRENT_WORKFLOW_EDITION.to_string();
        assert!(spec.validate().is_ok());
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
        spec.env_contract.num_envs = Some(4);
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
            spec.env_contract.num_envs = Some(4);
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
        assert!(value.get("envConnectTimeout").is_none());

        let parsed: RuntimeLimits = serde_json::from_value(json!({
            "envConnectTimeoutMs": 1,
            "modelConnectTimeoutMs": 2,
            "configureRouteTimeoutMs": 3,
            "envResetTimeoutMs": 4,
            "modelPredictTimeoutMs": 5,
            "envStepTimeoutMs": 6,
            "serviceCloseTimeoutMs": 7
        }))
        .unwrap();

        assert_eq!(parsed.env_connect_timeout, Duration::from_millis(1));
        assert_eq!(parsed.model_connect_timeout, Duration::from_millis(2));
        assert_eq!(parsed.configure_route_timeout, Duration::from_millis(3));
        assert_eq!(parsed.env_reset_timeout, Duration::from_millis(4));
        assert_eq!(parsed.model_predict_timeout, Duration::from_millis(5));
        assert_eq!(parsed.env_step_timeout, Duration::from_millis(6));
        assert_eq!(parsed.service_close_timeout, Duration::from_millis(7));
    }

    #[test]
    fn runtime_limits_reject_legacy_unsuffixed_fields() {
        let error = serde_json::from_value::<RuntimeLimits>(json!({
            "envConnectTimeout": 1
        }))
        .unwrap_err();

        assert!(error.to_string().contains("envConnectTimeout"));
    }
}
