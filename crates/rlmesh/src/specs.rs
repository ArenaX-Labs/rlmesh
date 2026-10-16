//! Python's `rlmesh.specs`: the read-only contract an env endpoint serves.

use std::fmt;

use rlmesh_spaces::AutoresetMode;

use crate::spaces::SpaceSpec;
use crate::value::meta_codec;
use crate::{Error, Metadata, Result, Value};

/// What an env endpoint serves: Python's `specs.EnvContract`.
///
/// Only the library builds one (an [`EnvServer`](crate::EnvServer) derives it
/// from the env; a remote client reads it from the handshake). Clones are
/// independent values.
#[derive(Clone)]
pub struct EnvContract {
    id: String,
    observation_space: SpaceSpec,
    action_space: SpaceSpec,
    render_mode: Option<String>,
    num_envs: u64,
    metadata: Option<Metadata>,
    autoreset_mode: AutoresetMode,
}

impl EnvContract {
    /// The env id (`"UnknownEnv-v1"` when the env sets none).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The space one observation belongs to.
    pub fn observation_space(&self) -> &SpaceSpec {
        &self.observation_space
    }

    /// The space one action belongs to.
    pub fn action_space(&self) -> &SpaceSpec {
        &self.action_space
    }

    /// The env's render mode, if any.
    pub fn render_mode(&self) -> Option<&str> {
        self.render_mode.as_deref()
    }

    /// The number of lanes the endpoint serves.
    pub fn num_envs(&self) -> u64 {
        self.num_envs
    }

    /// The env's metadata, tags included, if any.
    pub fn metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    /// Python's `.env` alias: a copy of this contract.
    pub fn env(&self) -> EnvContract {
        self.clone()
    }

    /// Python's `.spec` alias: a copy of this contract.
    pub fn spec(&self) -> EnvContract {
        self.clone()
    }

    /// Python's `to_dict()`: `id`, `render_mode`, `num_envs`, `metadata`, and
    /// each space as `{kind, shape, dtype, details}`.
    pub fn to_dict(&self) -> Metadata {
        let mut out = Metadata::new();
        out.insert("id", self.id.clone());
        out.insert(
            "render_mode",
            self.render_mode.clone().map_or(Value::None, Value::Text),
        );
        out.insert("num_envs", Value::Int(self.num_envs as i64));
        out.insert(
            "metadata",
            self.metadata.clone().map_or(Value::None, Value::Dict),
        );
        out.insert("observation_space", spec_dict(&self.observation_space));
        out.insert("action_space", spec_dict(&self.action_space));
        out
    }

    /// The contract a server derives (crate-private: only the library builds
    /// one).
    pub(crate) fn new(
        id: String,
        observation_space: SpaceSpec,
        action_space: SpaceSpec,
        render_mode: Option<String>,
        num_envs: u64,
        metadata: Option<Metadata>,
        autoreset_mode: AutoresetMode,
    ) -> EnvContract {
        EnvContract {
            id,
            observation_space,
            action_space,
            render_mode: render_mode.filter(|mode| !mode.is_empty()),
            num_envs,
            metadata,
            autoreset_mode,
        }
    }

    /// The facade contract for a wire contract; refuses one without spaces.
    pub(crate) fn from_wire(wire: &rlmesh_spaces::EnvContract) -> Result<EnvContract> {
        let space = |spec: &Option<rlmesh_spaces::SpaceSpec>, name: &str| {
            spec.clone()
                .map(SpaceSpec::from_wire)
                .ok_or_else(|| Error::Protocol(format!("env contract has no {name}")))
        };
        Ok(EnvContract::new(
            wire.id.clone(),
            space(&wire.observation_space, "observation_space")?,
            space(&wire.action_space, "action_space")?,
            Some(wire.render_mode.clone()),
            u64::from(wire.num_envs),
            wire.metadata.clone().map(meta_codec::metadata_from_wire),
            wire.autoreset_mode,
        ))
    }

    /// The wire contract this one serves as.
    pub(crate) fn to_wire(&self) -> Result<rlmesh_spaces::EnvContract> {
        Ok(rlmesh_spaces::EnvContract {
            id: self.id.clone(),
            action_space: Some(self.action_space.wire().clone()),
            observation_space: Some(self.observation_space.wire().clone()),
            metadata: self
                .metadata
                .as_ref()
                .map(meta_codec::metadata_to_wire)
                .transpose()?,
            render_mode: self.render_mode.clone().unwrap_or_default(),
            num_envs: u32::try_from(self.num_envs).unwrap_or(u32::MAX),
            autoreset_mode: self.autoreset_mode,
        })
    }

    /// The derived autoreset mode (not on Python's surface).
    pub(crate) fn autoreset_mode(&self) -> AutoresetMode {
        self.autoreset_mode
    }

    /// Construction step 7's lane comparison: the whole derived contract.
    pub(crate) fn same_lane_contract(&self, other: &EnvContract) -> bool {
        self.id == other.id
            && self.observation_space.to_space() == other.observation_space.to_space()
            && self.action_space.to_space() == other.action_space.to_space()
            && self.render_mode == other.render_mode
            && self.metadata == other.metadata
            && self.autoreset_mode == other.autoreset_mode
    }

    /// The same contract served at `num_envs` lanes.
    pub(crate) fn with_num_envs(mut self, num_envs: u64) -> EnvContract {
        self.num_envs = num_envs;
        self
    }
}

/// Python's `repr`: `EnvContract(id="CartPole-v1", num_envs=1)`.
impl fmt::Display for EnvContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EnvContract(id={:?}, num_envs={})",
            self.id, self.num_envs
        )
    }
}

impl fmt::Debug for EnvContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnvContract")
            .field("id", &self.id)
            .field("observation_space", &self.observation_space)
            .field("action_space", &self.action_space)
            .field("render_mode", &self.render_mode)
            .field("num_envs", &self.num_envs)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// A space spec as Python's `spec_to_dict`: `{kind, shape, dtype, details}`.
pub(crate) fn spec_dict(spec: &SpaceSpec) -> Value {
    match rlmesh_adapters::v1::space_spec_to_json(spec.wire()) {
        Ok(json) => json_to_value(json),
        Err(err) => Value::Text(err.to_string()),
    }
}

/// A JSON tree as a [`Value`] (objects become dicts, in the JSON's order).
pub(crate) fn json_to_value(json: serde_json::Value) -> Value {
    match json {
        serde_json::Value::Null => Value::None,
        serde_json::Value::Bool(flag) => Value::Bool(flag),
        serde_json::Value::Number(number) => match number.as_i64() {
            Some(int) => Value::Int(int),
            None => Value::Float(number.as_f64().unwrap_or(f64::NAN)),
        },
        serde_json::Value::String(text) => Value::Text(text),
        serde_json::Value::Array(items) => {
            Value::List(items.into_iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(map) => Value::Dict(
            map.into_iter()
                .map(|(key, value)| (key, json_to_value(value)))
                .collect(),
        ),
    }
}

/// A [`Value`] as JSON (tensors as nested lists, as the metadata codec
/// encodes them).
pub(crate) fn value_to_json(value: &Value) -> Result<serde_json::Value> {
    Ok(meta_to_json(&meta_codec::value_to_wire(value)?))
}

fn meta_to_json(value: &rlmesh_spaces::MetaValue) -> serde_json::Value {
    use rlmesh_spaces::MetaValue;
    match value {
        MetaValue::Null => serde_json::Value::Null,
        MetaValue::Bool(flag) => (*flag).into(),
        MetaValue::Int(int) => (*int).into(),
        MetaValue::Float(float) => serde_json::Number::from_f64(*float)
            .map_or(serde_json::Value::Null, serde_json::Value::Number),
        MetaValue::String(text) => text.clone().into(),
        MetaValue::Bytes(bytes) => bytes.to_vec().into(),
        MetaValue::List(items) => items.iter().map(meta_to_json).collect(),
        MetaValue::Map(map) => serde_json::Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), meta_to_json(value)))
                .collect(),
        ),
    }
}
