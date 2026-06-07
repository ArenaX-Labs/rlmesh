use std::time::Duration;

use rlmesh_proto::env::v1::EnvContract;
use rlmesh_proto::spaces::v1::SpaceSpec;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeSessionSpec {
    pub session_id: String,
    pub route_id: String,
    pub env_component_id: String,
    pub model_component_id: String,
    pub env_id: String,
    pub env_contract: EnvContract,
    pub num_envs: usize,
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
        if self.env_contract.observation_space.is_none() {
            return Err("runtime env_contract is missing observation_space".to_string());
        }
        if self.env_contract.action_space.is_none() {
            return Err("runtime env_contract is missing action_space".to_string());
        }
        if self.max_episodes == Some(0) {
            return Err("runtime max_episodes must be greater than zero when set".to_string());
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

    pub fn observation_space(&self) -> &SpaceSpec {
        self.env_contract
            .observation_space
            .as_ref()
            .expect("RuntimeSessionSpec validated observation_space")
    }

    pub fn action_space(&self) -> &SpaceSpec {
        self.env_contract
            .action_space
            .as_ref()
            .expect("RuntimeSessionSpec validated action_space")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

    use serde_json::json;

    use super::RuntimeLimits;

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
        assert!(value.get("telemetryWindow").is_none());

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
}
